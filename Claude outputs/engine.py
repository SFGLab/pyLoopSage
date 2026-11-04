"""OpenMM engine: relax a polymer under the loop constraints of each MC snapshot."""
import numpy as np
import openmm as mm
import openmm.unit as u
from openmm.app import DCDReporter, ForceField, PDBxFile, Simulation

from common.progress import stage_bar
from helpers.timing import Timer
from logger import get_logger
from molecular_dynamics import forces
from molecular_dynamics.cif_io import write_mmcif, write_psf
from molecular_dynamics.contacts import RunningMean, contact_map
from molecular_dynamics.result import MDResult
from molecular_dynamics.structures import make_structure

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

    def _platform(self):
        wanted = self.cfg.platform.lower()
        names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
        for name in names:
            if name.lower() == wanted:
                return mm.Platform.getPlatformByName(name)
        log.warning("Platform '%s' unavailable (have: %s): using CPU.", self.cfg.platform, ", ".join(names))
        return mm.Platform.getPlatformByName("CPU")

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

        integrator = mm.LangevinIntegrator(cfg.sim_temp * u.kelvin, cfg.friction / u.picosecond,
                                           cfg.integrator_step * u.femtosecond)
        if cfg.seed is not None:
            integrator.setRandomNumberSeed(cfg.seed)
        self.sim = Simulation(pdb.topology, self.system, integrator, self._platform())
        self.sim.context.setPositions(pdb.positions)
        self.platform_name = self.sim.context.getPlatform().getName()
        log.info("MD platform: %s", self.platform_name)
        self.sim.minimizeEnergy(tolerance=cfg.tolerance * u.kilojoule_per_mole / u.nanometer)
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
        coords, rg, epot, maps = np.zeros((nf, self.n, 3), np.float32), np.zeros(nf), np.zeros(nf), RunningMean()
        with Timer() as t, stage_bar("Molecular dynamics", nf) as advance:
            for k, i in enumerate(self.frames):
                self._update(i)
                self.sim.step(cfg.sim_step)
                state = self.sim.context.getState(getPositions=True, getEnergy=True)
                xyz = np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))
                coords[k], epot[k] = xyz, state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
                rg[k] = np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean())
                maps.add(contact_map(xyz, cfg.contact_radius))
                if cfg.save_ensemble:
                    self.dirs.ensemble.mkdir(exist_ok=True)
                    with open(self.dirs.ensemble / f"MDLE_{k + 1}.cif", "w") as f:
                        PDBxFile.writeFile(self.topology, state.getPositions(), f)
                advance()
        meta = dict(platform=self.platform_name, runtime_s=t.seconds, n_frames=nf, dt_fs=cfg.integrator_step,
                    steps_per_frame=cfg.sim_step, ps_per_frame=cfg.sim_step * cfg.integrator_step / 1000,
                    temperature_K=cfg.sim_temp, contact_radius_nm=cfg.contact_radius)
        return MDResult(coords, maps.mean, maps.std, rg, epot, self.frames, meta)


def run_md(cfg, geom, mc, dirs) -> MDResult:
    """Build and run the molecular dynamics stage."""
    engine = LoopMD(cfg, geom, mc, dirs)
    engine.build()
    return engine.run()
