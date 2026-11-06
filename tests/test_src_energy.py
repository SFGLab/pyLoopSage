"""Energy terms and the incremental-vs-total consistency of the MC sampler."""
from pathlib import Path

import numpy as np
import pytest

from monte_carlo.energy import E_cross, Kappa
from monte_carlo.energy import E_bind
from monte_carlo.energy import dE_bind_inner, E_bind_inner, E_fold, dE_fold


def _arr(*v):
    return np.array(v, dtype=np.int64)


def test_kappa_cases():
    assert Kappa(0, 5, 10, 15) == 0.0            # disjoint
    assert Kappa(0, 10, 5, 15) == 1.0            # crossing
    assert Kappa(0, 10, 10, 20) == 1.0           # shared endpoint
    assert Kappa(0, 20, 5, 15) == 0.0            # nested


def test_nested_loops_do_not_cross():
    assert E_cross(_arr(0, 5), _arr(20, 15), 1.0, 2, True, True) == 0.0


def test_fold_energy_longer_loops_are_lower():
    assert E_fold(_arr(0), _arr(50), -1.0) < E_fold(_arr(0), _arr(5), -1.0)


def test_dE_fold_matches_difference():
    old, new = E_fold(_arr(2), _arr(9), -1.3), E_fold(_arr(4), _arr(30), -1.3)
    assert dE_fold(-1.3, 2, 9, 4, 30) == pytest.approx(new - old)


def test_e_bind_sums_left_and_right():
    L, R = np.zeros(10), np.zeros(10)
    L[2], R[7] = 5.0, 3.0
    assert E_bind(L, R, _arr(2), _arr(7)) == 8.0


def test_bind_inner_rewards_spanning_covered_beads():
    cover = np.array([-1, -1, 1, 1, 1, -1, -1.0])
    P = np.r_[0.0, np.cumsum(cover)]
    inside, outside = E_bind_inner(P, -1.0, _arr(2), _arr(5)), E_bind_inner(P, -1.0, _arr(0), _arr(2))
    assert inside == -3 and outside == 2                    # covered span lowers the energy, loop-free span raises it
    assert dE_bind_inner(P, -1.0, 0, 2, 2, 5) == inside - outside


@pytest.mark.parametrize("extra", [[], ["--set", "epi_coeff=1.0", "--set", "epigenetic_mode=dynamic"],
                                   ["--set", "epi_coeff=1.0", "--set", "epigenetic_mode=static"],
                                   ["--binding-mode", "contrastive"]])
def test_mc_incremental_energy_matches_recomputed(fast_args, extra):
    """After sampling, the running energy must equal a fresh evaluation of the final state (all epigenetic modes)."""
    from monte_carlo.energy import total_energy
    from monte_carlo.sampler import build_params
    from monte_carlo.simulation import run_mc
    from preproc.loader import load_inputs
    from preproc.loader import build_geometry
    from settings.loader import load_config

    cfg = load_config(fast_args + extra)
    geom = build_geometry(cfg)
    inputs = load_inputs(cfg, geom)
    res = run_mc(cfg, inputs, geom)
    p = build_params(cfg, inputs, geom.n_beads, res.meta["n_lef"])
    last = res.n_samples - 1
    S = res.epi[:, last].astype(np.int64)
    e = total_energy(p, res.Ms[:, last].astype(np.int64), res.Ns[:, last].astype(np.int64), S)
    assert e == pytest.approx(res.Es[last], rel=1e-6, abs=1e-6)
    assert res.Es[-1] < res.Es[0] or extra                # energy went down (plain model)
    assert np.all(res.Ms < res.Ns) or np.all(res.Ms <= res.Ns)


def test_dynamic_epigenetic_incremental_matches_full():
    """dE for epigenetic-state changes and LEF moves equals the difference of full dynamic energies (incl. duplicate loops)."""
    from monte_carlo.energy import E_epi_dyn, dE_epi_dyn, dE_epi_move
    rng = np.random.default_rng(0)
    n, norm = 30, 2.0
    h = rng.normal(size=n)
    ms = np.array([3, 3, 10, 12, 20], np.int64)
    ns = np.array([9, 9, 18, 25, 22], np.int64)                  # first two LEFs share a loop
    S = rng.integers(0, 3, n).astype(np.int64)
    for _ in range(300):
        if rng.random() < 0.5:
            k, s = int(rng.integers(n)), int(rng.integers(3))
            d = dE_epi_dyn(S, ms, ns, h, norm, k, s)
            e0 = E_epi_dyn(S, ms, ns, h, norm); S2 = S.copy(); S2[k] = s
            assert np.isclose(d, E_epi_dyn(S2, ms, ns, h, norm) - e0)
            S = S2
        else:
            j = int(rng.integers(len(ms)))
            m2 = int(rng.integers(0, n - 3)); n2 = int(rng.integers(m2 + 2, n))
            if rng.random() < 0.3:
                m2, n2 = int(ms[1 - j if j < 2 else 0]), int(ns[1 - j if j < 2 else 0])   # land on an occupied loop
            d = dE_epi_move(S, ms, ns, norm, j, m2, n2)
            e0 = E_epi_dyn(S, ms, ns, h, norm); ms2, ns2 = ms.copy(), ns.copy(); ms2[j], ns2[j] = m2, n2
            assert np.isclose(d, E_epi_dyn(S, ms2, ns2, h, norm) - e0)
            ms, ns = ms2, ns2


def test_dynamic_epigenetic_coupling_structure():
    """All beads in one state: energy = -(backbone bonds + distinct LEF loops); no loops left -> backbone only."""
    from monte_carlo.energy import E_epi_dyn
    S, h = np.zeros(10, np.int64), np.zeros(10)
    ms, ns = np.array([1, 1, 4], np.int64), np.array([6, 6, 8], np.int64)
    assert E_epi_dyn(S, ms, ns, h, 1.0) == -(9 + 2)             # 9 backbone + 2 distinct loops
    assert E_epi_dyn(S, ms[:0], ns[:0], h, 1.0) == -9.0


def test_epigenetic_mode_is_config_only():
    from settings.loader import load_config
    import subprocess, sys
    cfg = load_config(["-f", "demo/data/demo_loops.bedpe", "-r", "chr1:77000000-95000000"])
    assert cfg.epigenetic_mode == "dynamic"
    out = subprocess.run([sys.executable, "src/run.py", "-h"], capture_output=True, text=True).stdout
    assert "epigenetic-mode" not in out and "epigenetic_mode" not in out


def _dense_J(n, ms, ns):
    """Reference coupling matrix: 1 on the backbone and on every current LEF loop, 0 elsewhere, zero diagonal."""
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = 1.0
    for m, k in zip(ms, ns):
        J[m, k] = J[k, m] = 1.0
    np.fill_diagonal(J, 0.0)
    return J


def test_dynamic_epigenetic_matches_dense_matrix_definition():
    """Fast incremental energies equal the energy computed from the explicit changing matrix J."""
    from monte_carlo.energy import E_epi, E_epi_dyn, dE_epi_dyn, dE_epi_move
    rng = np.random.default_rng(5)
    n, norm = 40, 1.7
    h = rng.normal(size=n)
    ms = rng.integers(0, 30, 6).astype(np.int64)
    ns = (ms + rng.integers(2, 9, 6)).astype(np.int64)
    ms[1], ns[1] = ms[0], ns[0]                                 # duplicate loop: J stays binary
    S = rng.integers(0, 3, n).astype(np.int64)
    assert np.isclose(E_epi_dyn(S, ms, ns, h, norm), E_epi(S, _dense_J(n, ms, ns), h, norm))
    for _ in range(200):
        j = int(rng.integers(len(ms)))
        m2 = int(rng.integers(0, n - 3)); n2 = int(rng.integers(m2 + 2, min(n, m2 + 12)))
        ms2, ns2 = ms.copy(), ns.copy(); ms2[j], ns2[j] = m2, n2
        ref = E_epi(S, _dense_J(n, ms2, ns2), h, norm) - E_epi(S, _dense_J(n, ms, ns), h, norm)
        assert np.isclose(dE_epi_move(S, ms, ns, norm, j, m2, n2), ref)          # the heat map changes: J_old -> J_new
        k, s = int(rng.integers(n)), int(rng.integers(3))
        S2 = S.copy(); S2[k] = s
        ref = E_epi(S2, _dense_J(n, ms, ns), h, norm) - E_epi(S, _dense_J(n, ms, ns), h, norm)
        assert np.isclose(dE_epi_dyn(S, ms, ns, h, norm, k, s), ref)
        ms, ns, S = ms2, ns2, S2


def test_epigenetic_model_off_by_default(tiny_args):
    """Default = pure loop extrusion: no epigenetic states are sampled and no epigenetic figures are made."""
    from settings.loader import load_config
    from monte_carlo.simulation import run_mc
    from preproc.loader import build_geometry, load_inputs
    cfg = load_config(tiny_args)
    assert cfg.epi_coeff == 0.0 and cfg.epigenetic_mode == "dynamic"
    geom = build_geometry(cfg)
    res = run_mc(cfg, load_inputs(cfg, geom), geom)
    assert not res.meta["epi_on"] and not res.epi.any() and res.acc[2, 0] == 0


def test_dynamic_epigenetic_model_end_to_end(tmp_path):
    """Demo run with the dynamic model: loop ends share a state more often than random pairs, report has the figures."""
    from analysis.ensemble import loop_state_agreement
    from pipeline.runner import run_pipeline
    from settings.loader import load_config
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(["-c", str(root / "demo/demo_epigenetic.ini"), "-f", str(root / "demo/data/demo_loops.bedpe"),
                       "-o", str(tmp_path), "--mc-steps", "5000", "--burnin", "1500",
                       "--set", f"comp_bed_file={root / 'demo/data/demo_compartments.bed'}"])
    ctx = run_pipeline(cfg)
    ag = loop_state_agreement(ctx.mc, cfg.n_epi_states)
    assert ctx.mc.meta["epi_on"] and ctx.mc.acc[2, 0] > 0
    assert ag["overall"] > ag["overall_base"] + 0.02                      # loops organise epigenetic domains
    html = Path(ctx.dirs.report).read_text()
    for key in ("Epigenetic model: domains and energy", "Epigenetic model: loops and epigenetic state", "average epigenetic state"):
        assert key in html


# ---- MD platform selection --------------------------------------------------

def _engine(platform, tmp_path, threads_source="default"):
    from types import SimpleNamespace
    from molecular_dynamics.engine import LoopMD
    e = LoopMD.__new__(LoopMD)
    e.cfg = SimpleNamespace(platform=platform, sim_temp=300, friction=1.0, integrator_step=10, seed=1, device="", threads=1,
                            source={"threads": threads_source})
    return e


def test_platform_candidates_prefer_gpu_then_cpu(tmp_path):
    mm = pytest.importorskip("openmm")
    have = {mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())}
    auto = _engine("auto", tmp_path)._candidates()
    assert auto[0] in ("CUDA", "OpenCL", "CPU") and "CPU" in auto and auto.index("CPU") < len(auto)
    order = [n for n in ("CUDA", "OpenCL", "CPU", "Reference") if n in have]
    assert auto == order                                                # fixed preference order
    assert _engine("cuda", tmp_path)._candidates() == order             # unavailable request falls back, never fails
    assert _engine("Reference", tmp_path)._candidates()[0] == "Reference"
    assert _engine("cpu", tmp_path)._candidates()[0] == "CPU"


def test_platform_falls_back_when_a_platform_fails_to_start(tmp_path, monkeypatch):
    mm = pytest.importorskip("openmm")
    from openmm import app
    import molecular_dynamics.engine as eng
    system = mm.System()
    system.addParticle(1.0)
    top = app.Topology()
    top.addAtom("A", app.Element.getBySymbol("C"), top.addResidue("R", top.addChain()))
    pdb = type("P", (), {"topology": top})()
    e = _engine("auto", tmp_path)
    e.system = system
    first = e._candidates()[0]
    real = eng.Simulation

    def flaky(topology, system, integrator, platform, props):
        if platform.getName() == first and len(e._candidates()) > 1:
            raise mm.OpenMMException("device not found\nsecond line")
        return real(topology, system, integrator, platform, props)

    monkeypatch.setattr(eng, "Simulation", flaky)
    if len(e._candidates()) < 2:
        pytest.skip("only one OpenMM platform installed")
    sim = e._simulation(pdb)
    assert sim.context.getPlatform().getName() == e._candidates()[1]            # moved on to the next platform


def test_cpu_platform_uses_all_cores_unless_threads_set(tmp_path):
    import os
    assert _engine("cpu", tmp_path)._cpu_threads() == (os.cpu_count() or 1)
    assert _engine("cpu", tmp_path, threads_source="cli")._cpu_threads() == 1
