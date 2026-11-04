"""LEF speeds: initial distributions, speed-weighted sliding, extrusion mode and velocity statistics."""
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.speed import lef_windows, msd, vacf
from monte_carlo.sampler import lef_speeds
from monte_carlo.simulation import run_mc
from preproc.loader import build_geometry, load_inputs
from settings.loader import load_config


def _cfg(kind, **kw):
    return SimpleNamespace(lef_speed_dist=kind, lef_speed_spread=0.5, lef_fast_fraction=0.2, lef_fast_ratio=5.0, **kw)


@pytest.mark.parametrize("kind", ["constant", "maxwell", "lognormal", "two_state"])
def test_speed_distributions_have_mean_one(kind):
    v = lef_speeds(_cfg(kind), 200, np.random.default_rng(0))
    assert v.mean() == pytest.approx(1.0) and (v > 0).all()
    assert (np.ptp(v) == 0) == (kind == "constant")


def test_two_state_has_two_levels_and_bad_values_raise():
    v = lef_speeds(_cfg("two_state"), 300, np.random.default_rng(1))
    assert len(np.unique(v)) == 2 and np.unique(v)[1] / np.unique(v)[0] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        lef_speeds(SimpleNamespace(lef_speed_dist="two_state", lef_speed_spread=0.5, lef_fast_fraction=2.0, lef_fast_ratio=5.0), 5, np.random.default_rng(0))


def _run(demo_bedpe, tmp_path, *extra):
    cfg = load_config(["-f", str(demo_bedpe), "-r", "chr1:77000000-95000000", "-n", "120", "-o", str(tmp_path), "--n-lef", "12",
                       "--method", "Metropolis", "--t-init", "1.0", "--mc-steps", "6000", "--sample-every", "5", "--burnin", "500",
                       "--no-run-MD", "--seed", "4", "--set", "bind_coeff=0.1", *extra])
    geom = build_geometry(cfg)
    return run_mc(cfg, load_inputs(cfg, geom), geom)


def test_faster_lefs_move_more(demo_bedpe, tmp_path):
    mc = _run(demo_bedpe, tmp_path, "--lef-speed", "two_state", "--lef-fast-ratio", "10", "--lef-fast-fraction", "0.5")
    v = np.array(mc.meta["lef_speed"])
    move = (np.abs(np.diff(mc.Ms, axis=1)) + np.abs(np.diff(mc.Ns, axis=1)))
    move = np.where(move < 20, move, 0).sum(axis=1)                       # sliding only, rebinding jumps removed
    assert move[v > 1].mean() > 1.5 * move[v < 1].mean()


def test_extrusion_mode_only_shrinks_by_rebinding(demo_bedpe, tmp_path):
    mc = _run(demo_bedpe, tmp_path, "--no-lef-rw")
    length = mc.Ns - mc.Ms
    d, new = np.diff(length, axis=1), length[:, 1:]
    assert (d > 0).any() and (new[d < 0] <= 12).all()            # a loop only gets shorter when it is reloaded as a minimal loop


class _Chain:
    """Synthetic chain: loop lengths follow a given trajectory."""
    def __init__(self, length):
        m = np.full_like(length, 50)
        self.Ms, self.Ns, self.meta = m, m + length, dict(mc_step=1)
        self.burnin_idx = 0

    def production(self, a):
        return a

    @property
    def chains(self):
        return [self]


def test_msd_exponent_ballistic_vs_diffusive():
    t = np.arange(400)
    ball = _Chain(np.tile(2 + t, (5, 1)))
    walk = _Chain((20 + np.cumsum(np.random.default_rng(0).choice([-1, 1], (40, 400)), axis=1)).clip(2))
    assert msd(ball)["alpha"] == pytest.approx(2.0, abs=0.3)
    assert msd(walk)["alpha"] == pytest.approx(1.0, abs=0.3)
    lag, c = vacf(ball)
    assert c[0] == pytest.approx(1.0)


def test_windows_do_not_cross_rebinding():
    ch = _Chain(np.array([[5] * 40 + [30] * 40]))
    ch.Ms = ch.Ms.copy()
    ch.Ms[0, 40:] = 90                                           # leg jump at t = 40: a rebinding
    ch.Ns = ch.Ms + np.array([[5] * 40 + [30] * 40])
    w = lef_windows(ch, window=8)
    assert len(w) == 10 and ((w.length == 5) | (w.length == 30)).all()


def test_speed_is_absolute_and_legs_move_at_plus_minus_v():
    from analysis.speed import velocities
    ch = _Chain(np.array([[4, 6, 8, 10, 8, 6]]))                      # right leg +1, left leg -1 (opening) then closing
    ch.Ms = 50 - (ch.Ns - 50) // 2
    ch.Ns = ch.Ms + np.array([[4, 6, 8, 10, 8, 6]])
    inputs = SimpleNamespace(loops=SimpleNamespace(L=np.zeros(100), R=np.zeros(100)))
    v = velocities(ch, inputs)
    assert (v["v"] >= 0).all() and np.allclose(v["v"], np.abs(v["vl"]) / 2) and np.allclose(v["v"], 1.0)


def test_free_energy_surface_and_interactive_html():
    from analysis.speed import free_energy_surface
    from viz.surface3d import surface_html
    rng = np.random.default_rng(0)
    x, y, F = free_energy_surface(rng.normal(5, 1, 500), rng.normal(2, 0.5, 500), bins=(12, 10))
    assert F.shape == (10, 12) and F.min() == pytest.approx(0.0, abs=0.5) and F.max() <= 7.0
    html = surface_html(x, y, F, ("a", "b"), uid="t")
    assert "<canvas id=\"t\"" in html and "{{" not in html


def test_free_energy_colors_never_white():
    from viz.surface3d import surface_html
    from viz.theme import FREE
    for t in np.linspace(0, 1, 50):
        r, g, b, _ = FREE(t)
        assert min(r, g, b) < 0.9                                 # every level stays visible on a white page
    page = surface_html(np.arange(4.0), np.arange(4.0), np.random.default_rng(0).random((4, 4)), ("a", "b"), "t")
    assert "'#ffffff'" not in page.split("stops=")[1].split("]")[0]
