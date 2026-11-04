"""Molecular dynamics output container (with save / load).

- Distances -> contact probabilities (Hi-C-like maps) from 3D coordinates.
- OpenMM engine: relax a polymer under the loop constraints of each MC snapshot.
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import openmm as mm
import openmm.unit as u
from openmm.app import DCDReporter, ForceField, PDBxFile, Simulation
from scipy.spatial.distance import pdist, squareform

from common.runtime import n_threads, stage_bar
from common.utils import Timer
from logger import get_logger
from molecular_dynamics import forces
from molecular_dynamics.structures import make_structure, write_mmcif, write_psf


# ---- Molecular dynamics output container (with save / load) ----------------

_ARRAYS = ("coords", "avg_map", "std_map", "rg", "epot", "frame_idx")
_DIST = ("avg_dist", "std_dist")


@dataclass
class MDResult:
    coords: np.ndarray     # (frames, beads, 3) positions in nm (float32)
    avg_map: np.ndarray    # mean contact strength (Gaussian kernel of distance) over frames
    std_map: np.ndarray    # std of the contact strength over frames
    rg: np.ndarray         # radius of gyration per frame, nm
    epot: np.ndarray       # potential energy per frame, kJ/mol
    frame_idx: np.ndarray  # MC snapshot index each frame was built from
    meta: dict = field(default_factory=dict)   # platform, runtime_s, dt_fs, steps per frame...
    avg_dist: np.ndarray = None    # mean pairwise distance over frames, nm
    std_dist: np.ndarray = None    # std of the pairwise distance over frames, nm

    def save(self, path) -> None:
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **{k: getattr(self, k) for k in _ARRAYS + _DIST})

    def save_maps(self, folder) -> None:
        """Average / std distance and strength matrices as .npy files."""
        Path(folder).mkdir(parents=True, exist_ok=True)
        for name, m in (("avg_distance", self.avg_dist), ("std_distance", self.std_dist),
                        ("avg_strength", self.avg_map), ("std_strength", self.std_map)):
            np.save(Path(folder) / f"md_{name}.npy", m)

    @classmethod
    def load(cls, path) -> "MDResult":
        z = np.load(path, allow_pickle=False)
        extra = {k: z[k] for k in _DIST if k in z.files}                 # absent in older runs
        return cls(**{k: z[k] for k in _ARRAYS}, meta=json.loads(str(z["meta"])), **extra)


# ---- Distances -> contact probabilities (Hi-C-like maps) from 3D coordinates ----

# contact kernels of distance d (nm) and contact radius r (nm)
KERNELS = {
    "gaussian": lambda d, r: np.exp(-d ** 2 / (2 * r ** 2)),
    "soft": lambda d, r: 1.0 / (1.0 + (d / r) ** 3),
    "binary": lambda d, r: (d <= r).astype(float),
    "exponential": lambda d, r: np.exp(-d / r),
    "power_law": lambda d, r: (d / r + 1e-3) ** -3.0,
}


def contact_map(coords, radius: float = 1.0, mode: str = "gaussian") -> np.ndarray:
    """N x N contact map of one frame (coords in nm)."""
    if not np.all(np.isfinite(coords)):
        raise ValueError("non-finite coordinates: the MD became unstable")
    return KERNELS[mode](squareform(pdist(coords)), radius)


class RunningMean:
    """Streaming mean and std of matrices (no need to keep every frame)."""

    def __init__(self):
        self.n, self.s, self.s2 = 0, None, None

    def add(self, x):
        if self.s is None:
            self.s, self.s2 = np.zeros_like(x, float), np.zeros_like(x, float)
        self.n += 1
        self.s += x
        self.s2 += x * x

    @property
    def mean(self):
        return self.s / self.n

    @property
    def std(self):
        return np.sqrt(np.maximum(self.s2 / self.n - self.mean ** 2, 0.0))


# ---- OpenMM engine: relax a polymer under the loop constraints of each MC snapshot ----

log = get_logger(__name__)


class LoopMD:
    """One MD frame per MC snapshot: update loop springs, integrate, record."""

    def __init__(self, cfg, geom, mc, dirs):
        self.cfg, self.mc, self.dirs, self.n = cfg, mc, dirs, geom.n_beads
        self.comps = cfg.do_compartments and mc.meta["epi_on"]
        if cfg.do_compartments and not self.comps:
            log.warning("do_compartments needs epi_coeff > 0 in the MC stage: compartments disabled.")
        first = mc.burnin_idx if cfg.md_skip_burnin else 0
        self.frames = np.arange(first, mc.n_samples)

    def _candidates(self):
        """Platform names to try, best first: the requested one (unless 'auto'), then CUDA, OpenCL, CPU, Reference."""
        have = {mm.Platform.getPlatform(i).getName().lower(): mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())}
        want = (self.cfg.platform or "auto").lower()
        if want != "auto" and want not in have:
            log.warning("Platform '%s' is not available here (have: %s): choosing automatically.", self.cfg.platform, ", ".join(have.values()))
        order = ([want] if want in have else []) + ["cuda", "opencl", "cpu", "reference"]
        return [have[n] for n in dict.fromkeys(order) if n in have]

    def _cpu_threads(self) -> int:
        """Threads for the OpenMM CPU platform: all cores unless the user set --threads."""
        explicit = getattr(self.cfg, "source", {}).get("threads", "default") != "default"
        return n_threads(self.cfg) if explicit else (os.cpu_count() or 1)

    def _simulation(self, pdb):
        """First platform that can actually create a context (a listed GPU platform may still fail); CPU runs in parallel."""
        cfg, errors = self.cfg, []
        for name in self._candidates():
            integrator = mm.LangevinIntegrator(cfg.sim_temp * u.kelvin, cfg.friction / u.picosecond, cfg.integrator_step * u.femtosecond)
            if cfg.seed is not None:
                integrator.setRandomNumberSeed(cfg.seed)
            props = {"Threads": str(self._cpu_threads())} if name == "CPU" else {}
            if cfg.device not in ("", None) and name in ("CUDA", "OpenCL"):
                props["DeviceIndex"] = str(cfg.device)
            try:
                sim = Simulation(pdb.topology, self.system, integrator, mm.Platform.getPlatformByName(name), props)
            except Exception as e:                                   # driver / device problems surface as OpenMMException
                errors.append(name)
                log.step("MD: platform %s could not start (%s)", name, str(e).strip().splitlines()[0][:80])
                continue
            plat = sim.context.getPlatform()
            try:
                detail = plat.getPropertyValue(sim.context, "DeviceName") if name != "CPU" else f"{plat.getPropertyValue(sim.context, 'Threads')} threads"
            except Exception:
                detail = ""
            log.step("MD platform: %s%s%s", name, f" ({detail})" if detail else "", f", after failing on {', '.join(errors)}" if errors else "")
            self.platform_detail = detail
            return sim
        raise RuntimeError("no OpenMM platform could be started")

    def _set_loops(self, i):
        """Replace the loop springs (always the last force) with those of snapshot i."""
        if self.loop_force is not None:
            self.system.removeForce(self.system.getNumForces() - 1)
        self.loop_force = forces.make_loop_force(self.mc.Ms[:, i], self.mc.Ns[:, i],
                                                 self.cfg.le_ff_length, self.cfg.le_ff_strength)
        self.system.addForce(self.loop_force)

    def build(self):
        """Initial structure, force field, minimisation."""
        cfg, d = self.cfg, self.dirs
        log.info("MD: building the initial %s structure and force field (%d beads)", cfg.initial_structure_type, self.n)
        write_mmcif(make_structure(self.n, cfg.initial_structure_type), d.structures / "init_structure.cif")
        write_psf(self.n, d.structures / "topology.psf")
        pdb = PDBxFile(str(d.structures / "init_structure.cif"))
        self.topology = pdb.topology
        self.system = ForceField(cfg.forcefield_path).createSystem(pdb.topology, nonbondedCutoff=1 * u.nanometer)
        self.ev = forces.add_excluded_volume(self.system, self.n, cfg.ev_ff_strength, cfg.ev_ff_power)
        forces.add_bonds(self.system, self.n)
        forces.add_angles(self.system, self.n, cfg.angle_ff_strength)
        if self.comps:
            self.comp = forces.add_compartments(self.system, self.mc.epi[:, self.frames[0]], np.sqrt(self.n) * 0.05)
        self.loop_force = None
        self._set_loops(self.frames[0])

        self.sim = self._simulation(pdb)
        self.sim.context.setPositions(pdb.positions)
        self.platform_name = self.sim.context.getPlatform().getName()
        kj = u.kilojoule_per_mole
        e0 = self.sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kj)
        log.info("MD: minimizing the energy of the initial structure")
        self.sim.minimizeEnergy(tolerance=cfg.tolerance * kj / u.nanometer)
        e1 = self.sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kj)
        log.step("MD: energy minimized (%.3g -> %.3g kJ/mol)", e0, e1)
        self.sim.reporters.append(DCDReporter(str(d.structures / "trajectory.dcd"), cfg.sim_step))

    def _update(self, i):
        """Move the context to MC snapshot i (loops, compartments, topoisomerase)."""
        cfg = self.cfg
        self._set_loops(i)
        self.sim.context.reinitialize(preserveState=True)
        if self.comps:
            for b in range(self.n):
                self.comp.setParticleParameters(b, [int(self.mc.epi[b, i])])
            self.comp.updateParametersInContext(self.sim.context)
        if cfg.ev_p > 0:
            forces.set_ev(self.ev, forces.ev_profile(self.n, cfg.ev_ff_strength, cfg.ev_p, cfg.continuous_top))
            self.ev.updateParametersInContext(self.sim.context)

    def run(self) -> MDResult:
        cfg, nf = self.cfg, len(self.frames)
        log.step("MD: simulating %d frames x %d steps (%.1f ps each)", nf, cfg.sim_step, cfg.sim_step * cfg.integrator_step / 1000)
        coords, rg, epot = np.zeros((nf, self.n, 3), np.float32), np.zeros(nf), np.zeros(nf)
        maps, dists = RunningMean(), RunningMean()
        with Timer() as t, stage_bar("Molecular dynamics", nf) as advance:
            for k, i in enumerate(self.frames):
                self._update(i)
                self.sim.step(cfg.sim_step)
                state = self.sim.context.getState(getPositions=True, getEnergy=True)
                xyz = np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))
                coords[k], epot[k] = xyz, state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
                rg[k] = np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean())
                d = squareform(pdist(xyz))
                dists.add(d)
                maps.add(KERNELS["gaussian"](d, cfg.contact_radius))
                if cfg.save_ensemble:
                    self.dirs.ensemble.mkdir(exist_ok=True)
                    with open(self.dirs.ensemble / f"MDLE_{k + 1}.cif", "w") as f:
                        PDBxFile.writeFile(self.topology, state.getPositions(), f)
                advance()
        log.info("MD: finished, %d structures%s", nf, " saved to the ensemble folder" if cfg.save_ensemble else "")
        meta = dict(platform=self.platform_name, platform_detail=getattr(self, "platform_detail", ""), runtime_s=t.seconds, n_frames=nf, dt_fs=cfg.integrator_step,
                    steps_per_frame=cfg.sim_step, ps_per_frame=cfg.sim_step * cfg.integrator_step / 1000,
                    temperature_K=cfg.sim_temp, contact_radius_nm=cfg.contact_radius)
        return MDResult(coords, maps.mean, maps.std, rg, epot, self.frames, meta, dists.mean, dists.std)


def run_md(cfg, geom, mc, dirs) -> MDResult:
    """Build and run the molecular dynamics stage."""
    engine = LoopMD(cfg, geom, mc, dirs)
    engine.build()
    res = engine.run()
    if cfg.save_metadata:
        res.save_maps(dirs.data / "metadata")
    return res
