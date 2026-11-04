"""Loop nesting: parent / child orders, summaries, figures and report entries."""
import numpy as np

from analysis.nesting import nesting, nesting_summary, order_counts


def _toy():
    ms = np.array([[0], [10], [20], [60], [60], [200]])
    ns = np.array([[100], [50], [30], [90], [90], [210]])
    return nesting(ms, ns)


def test_orders_and_duplicates():
    n = _toy()
    assert n.order[:, 0].tolist() == [0, 1, 2, 1, 1, 0]
    assert n.valid[:, 0].tolist() == [True, True, True, True, False, True]
    assert n.n_desc[:, 0].tolist() == [3, 1, 0, 0, 0, 0]


def test_parent_length_and_counts():
    n = _toy()
    assert np.isnan(n.parent_len[0, 0]) and n.parent_len[2, 0] == 40
    assert order_counts(n, 2)[:, 0].tolist() == [1, 1, 2, 1]        # orphan, parent, order 1, order 2+


def test_summary():
    s = nesting_summary([_toy()])
    assert s["max_order"] == 2 and abs(s["parent_fraction"] - 0.2) < 1e-9 and abs(s["orphan_fraction"] - 0.2) < 1e-9
    assert abs(s["mean_child_order"] - 4 / 3) < 1e-9


# ---- Real run: invariants, figures, report entries ----

import pytest

from analysis.nesting import production_nesting, pooled
from monte_carlo.simulation import run_mc
from preproc.loader import build_geometry, load_inputs
from reporting.explanations import EXPLAIN
from reporting.report import REGISTRY
from settings.loader import load_config
from viz.nesting_figs import nesting_figure, parents_children_figure


@pytest.fixture(scope="module")
def run(demo_bedpe, tmp_path_factory):
    out = tmp_path_factory.mktemp("nest")
    cfg = load_config(["-f", str(demo_bedpe), "-r", "chr1:77000000-95000000", "-n", "120", "-o", str(out), "--n-lef", "12",
                       "--method", "Metropolis", "--t-init", "1.0", "--mc-steps", "6000", "--sample-every", "5", "--burnin", "500",
                       "--no-run-MD", "--seed", "4", "--set", "bind_coeff=0.1"])
    geom = build_geometry(cfg)
    res = run_mc(cfg, load_inputs(cfg, geom), geom)
    return res, geom, production_nesting(res)


def test_real_run_invariants(run):
    _, _, parts = run
    for p in parts:
        assert (p.order[p.valid] >= 0).all()
        assert ((p.order == 0) == np.isnan(p.parent_len))[p.valid].all()        # parents have no enclosing loop
        assert (p.parent_len[p.valid & (p.order > 0)] > p.length[p.valid & (p.order > 0)] - 1e-9).all()
        assert (p.n_desc[p.valid] <= p.valid.sum(0).max()).all()
    o, _ = pooled(parts, "order")
    assert (o == 0).any()                                                      # there is always an outermost loop


def test_figures_registered_with_text(run):
    res, geom, parts = run
    REGISTRY.figs.clear() if hasattr(REGISTRY, "figs") else None
    nesting_figure(res, parts)
    parents_children_figure(res, parts, geom)
    for key in ("loop_nesting", "loop_parents_children"):
        assert EXPLAIN[key]["title"] and EXPLAIN[key]["explain"] and EXPLAIN[key]["section"] == "monte_carlo"
