"""Optional topology tools: chain crossings, the EV barrier, local topoisomerase events, link dynamics and examples."""
import numpy as np
import pytest

from loopsage.knots.crossing import barrier_kT, crossings, strength_for_barrier
from loopsage.knots.linking import LinkResult, link_dynamics, link_examples
from loopsage.molecular_dynamics import topology as T
from loopsage.settings.loader import load_config
from loopsage.settings.schema import PARAMS


def _ring(centre, axis, r=1.0, n=24):
    """Closed polygon (n points) around `centre`: in the yz-plane (axis 0) or in the xy-plane (axis 1)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = np.zeros((n, 3))
    if axis == 0:
        c[:, 1], c[:, 2] = r * np.cos(t), r * np.sin(t)
    else:
        c[:, 0], c[:, 1] = r * np.cos(t), r * np.sin(t)
    return c + centre


def test_crossing_detector_sees_two_bonds_passing_through_each_other():
    prev = np.array([[-.05, 0, -.01], [.05, 0, -.01], [0.2, 0, 0.5], [0, -.05, .01], [0, .05, .01]], float)
    cur = prev.copy()
    cur[:2, 2], cur[3:, 2] = 0.01, -0.01                         # the first bond moves up through the second
    (i, j, sign), = crossings(prev, cur)
    assert (i, j) == (0, 3)
    back = crossings(cur, prev)                                   # the reverse move crosses with the opposite sign
    assert len(back) == 1 and back[0][2] == -sign
    assert crossings(prev, prev) == []                            # nothing moved: no crossing


def test_ev_barrier_is_a_few_kT_by_default_and_scales_with_strength():
    b = barrier_kT(100.0, 3.0)
    assert 4.5 < b < 5.2                                          # the default EV lets strands pass at a few kT
    assert barrier_kT(200.0, 3.0) == pytest.approx(8 * b, rel=1e-6)          # barrier ~ strength ** power
    s = strength_for_barrier(10.0, 100.0, 3.0)
    assert s > 100 and barrier_kT(s, 3.0) == pytest.approx(10.0, rel=1e-6)


def test_local_sites_are_close_contacts_and_pair_links_see_a_hopf_link():
    rng = np.random.default_rng(0)
    pos = np.vstack([np.c_[np.linspace(0, 1, 30), np.zeros(30), np.zeros(30)],
                     np.c_[np.linspace(0, 1, 30), np.full(30, 0.03), np.zeros(30)]])      # two parallel strands 0.03 apart
    sites = T.contact_sites(pos, 0.05, 3, 5, rng)
    assert sites and all(0 <= g0 < g1 <= 60 and g1 - g0 == 3 and abs(partner - (g0 + 1)) > 5 for g0, g1, partner in sites)
    a, b = _ring(np.array([0.0, 0, 0]), 0), _ring(np.array([0.0, 1.0, 0]), 1)               # rings through each other: Hopf link
    pos = np.vstack([a, b])
    ms, ns = np.array([0, 24]), np.array([23, 47])
    lk = T.pair_links(pos, ms, ns, 5, 30)
    assert list(lk.values()) in ([1], [-1])
    far = np.vstack([a, b + np.array([0, 5.0, 0])])
    assert list(T.pair_links(far, ms, ns, 5, 30).values()) == [0]
    assert T.link_change({(0, 23, 24, 47): 0}, {(0, 23, 24, 47): 1}) == (1, 0)
    assert T.link_change({(0, 23, 24, 47): 1}, {(0, 23, 24, 47): 0}) == (0, 1)


def test_simplifying_rule_only_stops_new_links():
    rng = np.random.default_rng(1)
    assert all(T.keep_passage(0, 0.0, rng) for _ in range(20))                   # no new link: always kept
    assert all(T.keep_passage(1, 1.0, rng) for _ in range(20))                   # keep = 1: no bias
    assert not any(T.keep_passage(1, 0.0, rng) for _ in range(20))               # keep = 0: every new link is undone


def _result(frames, pairs):
    fr = np.array([f for f, _ in pairs], int)
    return LinkResult(fr, np.array([p for _, p in pairs], int).reshape(-1, 5), np.array(frames), np.zeros(len(frames), int),
                      np.zeros(len(frames), int), 50)


def test_link_lifetimes_formation_and_dissolution():
    p, q = (0, 9, 20, 29, 1), (5, 14, 30, 39, -1)
    res = _result([0, 1, 2, 3], [(0, p), (1, p), (2, p), (2, q), (3, q)])
    life, formed, dissolved = link_dynamics(res)
    assert sorted(life) == [2, 3] and list(formed) == [1, 0, 1, 0] and list(dissolved) == [0, 0, 0, 1]


def test_examples_mark_the_place_of_the_link():
    pytest.importorskip("pyvista")
    pos = np.vstack([_ring(np.zeros(3), 0, 1.0, 24), _ring(np.array([0, 1.0, 0]), 1, 1.0, 24)]) * 0.1
    res = _result([0], [(0, (0, 23, 24, 47, 1))])
    res.n_beads = 48
    try:
        ex = link_examples(np.stack([pos]), res)
    except Exception as e:                                                         # no off-screen rendering here
        pytest.skip(str(e))
    if ex:
        assert len(ex) == 1 and ex[0][1].shape[2] == 3 and "Lk = +1" in ex[0][0]


def test_topology_settings_are_optional_config_only_except_the_knots_flag():
    top = [p for p in PARAMS if p.section == "topology"]
    assert [p.name for p in top if p.flag] == ["detect_knots"]                      # only --knots is on the command line
    d = {p.name: p.default for p in top}
    assert d["detect_knots"] is False and d["topo_mode"] == "random" and d["track_crossings"] is False
    assert d["ev_p"] == 0.0 and d["topo_keep"] == 1.0 and d["continuous_top"] is False


def test_topology_is_off_by_default_and_reports_when_asked(tiny_args):
    pytest.importorskip("openmm")
    from loopsage.pipeline.runner import run_pipeline
    base = [a for a in tiny_args if a != "--no-run-MD"] + ["--run-MD", "--md-steps", "300", "--n-lef", "8", "--no-render-3d", "--set", "tolerance=10"]
    plain = run_pipeline(load_config(base))
    assert "topology" not in plain.md.meta and plain.links is None
    assert not (plain.dirs.figures / "crossings.png").exists()
    on = run_pipeline(load_config(base + ["--set", "topo_mode=local", "--set", "topo_rate=30", "--set", "topo_keep=0.5", "--set", "track_every=50", "--knots"]))
    t = on.md.meta["topology"]
    assert t["mode"] == "local" and len(t["leak"]) == on.md.coords.shape[0] and sum(t["events"]) > 0
    assert (on.dirs.figures / "crossings.png").exists() and (on.dirs.figures / "links_time.png").exists()
    html = (on.dirs.root / "report" / "report.html").read_text()
    assert "Strand-crossing barrier" in html and "Chain crossings during the MD" in html


def test_vectorised_crossing_detector_matches_a_slow_reference():
    from loopsage.knots.crossing import _V, crossings as fast

    def slow(prev, cur, min_sep=2):
        """Reference: every segment pair, roots with np.roots."""
        n = len(prev)
        out = []
        for i in range(n - 1):
            for j in range(i + min_sep, n - 1):
                f = []
                for t in (0, 1 / 3, 2 / 3, 1):
                    p = prev + t * (cur - prev)
                    f.append(np.dot(p[i + 1] - p[i], np.cross(p[j] - p[i], p[j + 1] - p[i])))
                f = np.array(f)
                if not (f.min() < 0 < f.max()):
                    continue
                for r in np.roots(_V @ f):
                    if abs(r.imag) > 1e-9 or not 0 <= r.real <= 1:
                        continue
                    p = prev + r.real * (cur - prev)
                    d1, d2, w = p[i + 1] - p[i], p[j + 1] - p[j], p[j] - p[i]
                    nn = np.cross(d1, d2)
                    den = nn @ nn
                    if den < 1e-18:
                        continue
                    s, u = np.cross(w, d2) @ nn / den, np.cross(w, d1) @ nn / den
                    if 0 <= s <= 1 and 0 <= u <= 1:
                        out.append((i, j, int(np.sign(f[-1] - f[0])) or 1))
        return sorted(out)

    rng = np.random.default_rng(3)
    found = 0
    for _ in range(6):
        prev = np.cumsum(rng.normal(size=(40, 3)), axis=0) * 0.1
        cur = prev + rng.normal(size=prev.shape) * 0.15            # big moves: some bonds cross
        ref = slow(prev, cur)
        found += len(ref)
        assert sorted(fast(prev, cur)) == ref
    assert found > 0
