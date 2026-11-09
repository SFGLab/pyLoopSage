"""sensitivity_report.html: one self-contained page that combines all runs of a scan."""
import base64
import datetime
import html
from pathlib import Path

import numpy as np
import pandas as pd

from loopsage.sensitivity import analysis as A
from loopsage.sensitivity import metrics as M
from loopsage.sensitivity import phases as PH
from loopsage.sensitivity import plots as P

esc = html.escape
HTML_DIR = Path(__file__).resolve().parents[1] / "html"
EXTRA_CSS = """
.insight{margin:0;padding-left:20px}.insight li{margin:7px 0}
.warn{background:#fff6ea;border-left:5px solid #f58a3c;border-radius:12px;padding:10px 16px;margin:0 0 14px;font-weight:600}
.note{background:#f1eefb;border-left:5px solid #6a3de8;border-radius:12px;padding:10px 16px;margin:0 0 14px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.pill{display:inline-block;background:#f1eefb;color:#4b2bb8;border-radius:999px;padding:1px 10px;font-size:.8rem;font-weight:800;margin-right:6px}
h3 small{color:#5a5a66;font-weight:600;font-size:.82rem;margin-left:8px}
"""
GROUPS = [
    ("loops", "Loops and folding", "How long the loops are and how much of the polymer they cover.", ["mean_loop_length", "longest_loop", "unfolded_fraction", "n_loops"]),
    ("nesting", "Nesting: parents and children", "Loops can sit inside each other. A parent holds loops and sits in none; a child sits inside another loop.", ["n_parents", "n_children", "n_orphans", "nesting_depth"]),
    ("binding", "Binding at CTCF sites", "How often the legs of the rings rest on CTCF sites.", ["bound_fraction", "anchored_loop_fraction"]),
    ("energy", "Energy", "The effective energy of the loop configurations.", ["energy_total", "folding_energy", "binding_energy", "crossing_energy"]),
    ("structure", "3D structure", "Size and shape of the polymer built from the loops (molecular dynamics).", list(M.MD_METRICS)),
    ("topology", "Topology: links and chain crossings", "Whether loops are linked in 3D, and whether the chain passes through itself.", list(M.TOPO_METRICS)),
]
EXPLAIN = {
    "traces": "Each line is the average over the runs that share one value of the input (the other inputs and the replicates are pooled); the band is the middle 70% of those runs. The tinted start is the burn-in, which is discarded. A curve that is flat after the burn-in has equilibrated; one that keeps drifting needs more steps.",
    "response": "Dots are single runs (production mean after the burn-in). The black line is the mean over runs and the band its 95% interval. A flat line means the metric does not care about this input; a steep one means it matters. With several inputs the others are averaged out.",
    "heatmap": "The mean of the metric for every combination of the two inputs (the rest averaged out). Colour runs from low (light) to high (warm). Look for diagonal bands (the two inputs trade off) and for corners where the metric changes abruptly.",
    "lines": "The same data as the heat map as curves: the metric against the first input, one curve per value of the second. Parallel curves mean the two inputs act independently; curves that cross or fan out mean they interact.",
    "main": "One panel per input: the mean of the metric in bins of that input, with the other inputs averaged out (main effect). The panel with the largest vertical change is the most influential input; all panels share the vertical axis.",
    "oat": "One-at-a-time scan: each panel changes one input while all the others stay at their baseline. This shows the local effect of each input, but not how inputs interact.",
    "importance": "Each cell is the share of the variance of the metric that is explained by that input alone (first-order effect, eta squared). The values of one column do not add to 100%: what is missing is interaction between inputs plus noise.",
    "sobol": "Sobol indices. S1 (violet) is the share of the variance explained by the input alone; ST (black) includes all its interactions with other inputs. ST much larger than S1 means the input acts mostly through combinations with others. Bars show 90% bootstrap intervals.",
    "tradeoff": "Each dot is one parameter point (mean over replicates), coloured by the most influential input. A diagonal cloud means the two metrics move together; a curve that bends back means a trade-off.",
    "profiles": "How the rings are distributed along the polymer. Each row is one value of the input. Dotted lines mark the CTCF sites (red: forward, blue: reverse). Peaks at the sites mean that legs rest there; a flat profile means loops are spread everywhere.",
    "coverage": "The share of snapshots in which a bead lies inside a loop. High everywhere means the polymer is fully folded; dips show loop-free stretches.",
    "sizes": "The distribution of loop lengths in logarithmic bins. A shift to the right means longer loops; a second bump means two populations of loops (for example short anchored loops and long extruded ones).",
    "convergence": "Left: effective number of independent samples in each run (the red line marks 30; below it the mean is poorly determined). Middle: how much the mean differs between the first and second half of the production run (red line: 10%). Right: the spread of the means between replicates.",
    "phase": "Top: the order parameter (mean over replicates, line; single runs, dots). The shaded band is the transition interval and the dashed red line its centre. Middle: the fluctuations of the order parameter in time (susceptibility); a peak at the transition is the mark of a continuous transition. Bottom: the autocorrelation time; a peak means critical slowing down, the system takes much longer to forget its state.",
    "bifurcation": "Violet dots are snapshots of the order parameter during the production run, at each value of the control parameter; coloured lines connect the mean of each replicate. One cloud that moves smoothly means a continuous change. Two separate clouds at one value, or replicates that follow different lines, mean two coexisting states: a bifurcation or a first-order transition.",
    "distribution": "Histograms of the order parameter at values of the control parameter across the transition (bottom: low value, top: high value). One peak that slides is a continuous change; a second peak that appears and grows while the first fades is a discontinuous one, and at the transition both exist at the same time.",
    "phasemap": "Left: the mean of the order parameter on the plane of two inputs. Right: its fluctuation in time. Red diamonds mark the detected transition along the horizontal input. A ridge of high fluctuation that follows the transition is its phase boundary: it shows how the transition point moves with the other input.",
    "binder": "The Binder cumulant U4 measures the shape of the fluctuations: 2/3 for a sharply ordered state, 0 for Gaussian noise. For a continuous transition the curves of different system sizes cross at one point, the critical value of the control parameter. Curves that do not cross suggest a crossover or a first-order transition.",
    "leak": "Crossing leaks counted during the dynamics, against the strand-crossing barrier of the excluded volume. A leak means the chain passed through itself even though no topoisomerase was active; barriers of about 10 kT or more keep the topology intact.",
}


def _img(b: bytes, alt="") -> str:
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{base64.b64encode(b).decode()}">'


def _card(title, figs, how, sub="", intro="") -> str:
    """A card with a short intro, one or more figures and a 'how to read' box."""
    body = "".join(f"<figure>{_img(f, title)}</figure>" for f in figs)
    grid = f'<div class="grid2">{body}</div>' if len(figs) > 1 else body
    lead = f"<p class='intro' style='margin-left:0'>{esc(intro)}</p>" if intro else ""
    return (f'<div class="card"><h3>{esc(title)}<small>{esc(sub)}</small></h3>{lead}{grid}'
            f'<details open><summary>How to read this</summary><p>{how}</p></details></div>')


def _table(cols, rows) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _g(v):
    return v if isinstance(v, str) else f"{v:.3g}"


class Builder:
    def __init__(self, res):
        self.res, self.spec, self.df = res, res.spec, res.table
        self.axes, self.k = res.design.axes, len(res.design.axes)
        self.kind = res.design.kind
        self.imp = A.importance(res) if len(self.df) >= 8 else pd.DataFrame(columns=["metric", "axis", "eta2", "rho", "rel_range"])
        self.sob = A.sobol(res) if self.kind == "sobol" and len(self.df) else None
        names = {a.label: a for a in self.axes}
        px = self.spec.plots.get("x")
        self.prim = names.get(px) or next((a for a in self.axes if a.numeric), self.axes[0])
        sy = self.spec.plots.get("y")
        self.sec = names.get(sy) or next((a for a in self.axes if a is not self.prim), None)
        self.sections = []

    # ---- one metric -----------------------------------------------------------------------------------
    def top_axes(self, metric, n=2):
        d = self.imp[self.imp.metric == metric].sort_values("eta2", ascending=False)
        out = [a for a in d.axis if a in {x.label for x in self.axes}][:n]
        return [next(x for x in self.axes if x.label == a) for a in out] or self.axes[:n]

    def metric_card(self, m, headline=True) -> str:
        res, df, p, s = self.res, self.df, self.prim, self.sec
        title = A.label(m)
        helptxt = M.METRICS[m].help if m in M.METRICS else (M.MD_METRICS.get(m) or M.TOPO_METRICS.get(m) or ("", "", ""))[2]
        figs, how = [], []
        if m not in df.columns or df[m].notna().sum() < 2:
            return ""
        if self.kind == "oat":
            figs.extend(P.main_effects(res, m, oat=True)); how.append(EXPLAIN["oat"])
        elif self.kind in ("random", "sobol") or self.k >= 3:
            figs.extend(P.main_effects(res, m)); how.append(EXPLAIN["main"])
            a, b = (self.top_axes(m, 2) + [None, None])[:2]
            if self.kind == "grid" and b is not None and a.numeric and b.numeric:
                figs.append(P.heatmap(res, m, a.label, b.label)); how.append(EXPLAIN["heatmap"])
        elif self.k == 2 and s is not None:
            figs.append(P.heatmap(res, m, p.label, s.label)); figs.append(P.lines_by(res, m, p.label, s.label)); how += [EXPLAIN["heatmap"], EXPLAIN["lines"]]
        else:
            if m in M.METRICS and headline:
                figs.append(P.traces(res, m, p.label)); how.append(EXPLAIN["traces"])
            figs.append(P.response(res, m, p.label)); how.append(EXPLAIN["response"])
        return _card(title, figs, " ".join(dict.fromkeys(how)), intro=helptxt)

    # ---- sections ---------------------------------------------------------------------------------------------
    def overview(self):
        res, spec, df = self.res, self.spec, self.df
        n_ok = len(df)
        stats = [("Design", spec.design), ("Parameter points", res.design.n_points), ("Replicates", spec.replicates),
                 ("Runs", f"{n_ok} of {len(res.tasks)}"), ("From cache", res.cache_hits), ("Workers", res.workers),
                 ("Wall time", f"{res.wall_s:.0f} s" if res.wall_s < 600 else f"{res.wall_s / 60:.1f} min"),
                 ("Polymer", f"{spec.system['n_beads']} beads, {spec.system['n_ctcf']} CTCF"), ("3D step", "on" if spec.md else "off")]
        cards = "".join(f'<div class="stat"><small>{esc(k)}</small><span>{esc(str(v))}</span></div>' for k, v in stats)
        axes = _table(["Input", "Values scanned", "Type"], [[a.label, (f"{fmt_range(a)}"), a.kind] for a in self.axes])
        warn = "".join(f'<div class="warn">{esc(w)}</div>' for w in spec.warnings + self.res.notes)
        if res.failures:
            warn += f'<div class="warn">{len(res.failures)} run(s) failed and are left out; see the quality section.</div>'
        intro = ("<p class='intro'>Each run is a small, independent loop-extrusion simulation of a synthetic polymer. The report combines all of them: "
                 "how the quality measures respond to the inputs you scanned, which input matters most, and whether the results can be trusted. "
                 "Nothing is saved per run; the combined tables are written next to this page.</p>")
        return f'<section id="overview"><h2>Overview</h2>{intro}{warn}<div class="cards">{cards}</div><div class="card"><h3>Scanned inputs</h3>{axes}</div></section>'

    def glance(self):
        res, imp = self.res, self.imp
        if imp.empty:
            return ""
        parts = []
        sents = A.insights(res, imp)
        if self.k >= 2 or self.kind in ("random", "sobol", "oat"):
            parts.append(_card("Which input matters for which output", [P.importance_map(imp)], EXPLAIN["importance"], "first-order effects"))
        if self.sob is not None and len(self.sob):
            head = [m for m in A.metric_cols(self.df)][:6]
            parts.append(_card("Sobol sensitivity indices", P.sobol_bars(self.sob, head), EXPLAIN["sobol"], "alone and with interactions"))
        lis = "".join(f"<li>{t}</li>" for _, t in sents)
        parts.insert(0, f'<div class="card"><h3>What the scan says</h3><ul class="insight">{lis}</ul></div>')
        if self.k == 1:
            rows = []
            a = self.axes[0]
            for m in A.metric_cols(self.df):
                g = A.agg_simple(self.df, m, a.label) if hasattr(A, "agg_simple") else P.agg(self.df, m, a.label)["mean"]
                el = A.elasticity(g.index.astype(float), g.values) if a.numeric else np.nan
                rows.append([A.label(m), _g(g.iloc[0]), _g(g.iloc[-1]), f"{(g.iloc[-1] - g.iloc[0]) / (abs(g.iloc[0]) + 1e-12):+.0%}", "" if np.isnan(el) else f"{el:+.2f}"])
            parts.append('<div class="card"><h3>Effect across the scanned range</h3>' + _table(["Metric", f"at {a.label} = {_g(g.index[0])}", f"at {a.label} = {_g(g.index[-1])}", "change", "elasticity"], rows)
                         + "<p class='caption'>Elasticity is the percent change of the metric per percent change of the input (log-log slope): 1 means proportional, 0 means no effect.</p></div>")
        return '<section id="glance"><h2>At a glance</h2><p class="intro">The short version: what responds to what.</p>' + "".join(parts) + "</section>"

    def group_section(self, gid, title, intro, names):
        cols = [m for m in names if m in self.df.columns]
        extra = [m for m in A.metric_cols(self.df) if gid == "other" and not any(m in g[3] for g in GROUPS)]
        cols = cols + extra
        if not cols:
            return ""
        head = [m for m in cols][:6]
        cards = []
        if gid == "topology":
            cards.append(self.topology_intro())
        for m in cols:
            c = self.metric_card(m, headline=m in head)
            if c:
                cards.append(c)
        if gid == "loops":
            cards += self.profile_cards()
        if gid == "binding":
            cards += self.tradeoff_cards()
        if gid == "topology":
            cards += self.topology_extra()
        return f'<section id="{gid}"><h2>{esc(title)}</h2><p class="intro">{esc(intro)}</p>{"".join(cards)}</section>' if cards else ""

    def profile_cards(self):
        res, p = self.res, self.prim
        if len([r for r in res.records.values() if r["ok"]]) < 2 or p.name == "system.n_beads":
            return []
        sites = None
        if p.name not in ("system.n_ctcf", "system.n_beads", "system.layout", "system.orientation", "system.strength"):
            from loopsage.sensitivity.runner import make_system
            box, _ = make_system(self.spec.system)
            sites = [(pos / box.bp_per_bead, "forward" if q < 0.5 else "reverse") for pos, _, q in box.ctcf_sites.itertuples(index=False)]
        try:
            return [_card(f"Where the legs sit, by {p.label}", [P.profiles(res, p.label, "leg_density", sites)], EXPLAIN["profiles"], "per-bead occupancy"),
                    _card(f"Loop coverage, by {p.label}", [P.profiles(res, p.label, "coverage", sites)], EXPLAIN["coverage"], "share of snapshots inside a loop"),
                    _card(f"Loop-length distribution, by {p.label}", [P.loop_sizes(res, p.label)], EXPLAIN["sizes"])]
        except (StopIteration, ValueError):
            return []

    def tradeoff_cards(self):
        cols = set(self.df.columns)
        pairs = [("mean_loop_length", "bound_fraction"), ("n_parents", "n_children")]
        pairs = [p for p in pairs if set(p) <= cols]
        if len(self.res.design.points) < 6 or not pairs:
            return []
        col = self.top_axes(pairs[0][0], 1)[0].label
        return [_card("Trade-offs between outputs", [P.tradeoff(self.res, a, b, col) for a, b in pairs], EXPLAIN["tradeoff"], f"coloured by {col}")]

    def topology_intro(self):
        return ('<div class="note"><b>What is measured.</b> Every 3D frame is checked for pairs of separate loops that are <b>linked</b>: closing each loop '
                'by the chord between its anchors, the Gauss linking number Lk of the pair is computed, and |Lk| &ge; 1 counts as a link. Linked loops cannot be pulled apart without '
                'a strand passing through another, which is what topoisomerase II does. <b>Leaks</b> count strand crossings that happen although the excluded volume is on. '
                'Linked pairs are <i>rare events</i>, so their share is pooled over all frames and replicates (the number of tested pairs is given below); '
                'use more 3D frames and replicates when the counts are small.</div>')

    def topology_extra(self):
        df, out = self.df, []
        if "ev_barrier" in df and df["ev_barrier"].nunique() >= 3 and "leak_rate" in df:
            out.append(_card("Crossing leaks and the excluded-volume barrier", [P.leak_vs_barrier(self.res)], EXPLAIN["leak"]))
        if "link_den" in df:
            by = [a.label for a in self.axes]
            g = df.groupby(by)[["link_num", "link_den"]].sum().reset_index()
            rows = [[*(_g(v) for v in r[by]), int(r.link_den), int(r.link_num), f"{r.link_num / r.link_den:.2%}" if r.link_den else "n/a"] for _, r in g.iterrows()]
            low = (g.link_den < 30).mean()
            note = (f"<p class='caption'>For {low:.0%} of the points fewer than 30 disjoint pair-frames were tested: the share of linked pairs is imprecise there. Increase md_frames, replicates or n_lef.</p>" if low > 0.3 else "")
            out.append(f'<div class="card"><h3>Counts behind the linked fraction</h3>{_table(by + ["pairs tested", "linked", "share"], rows[:60])}{note}</div>')
        return out

    def phases(self):
        """The 'Phase transitions and bifurcations' section."""
        res, df = self.res, self.df
        axes = PH.numeric_axes(res)
        defs = ("<div class='note'><b>Order parameter</b>: a metric that summarises the state, for example the mean loop length. <b>Control parameter</b>: the input that is varied. "
                "A <b>phase transition</b> is a sharp change of the order parameter along the control parameter. It is <b>continuous</b> if the order parameter changes smoothly but fluctuates strongly and relaxes slowly at the transition "
                "(critical slowing down), and <b>discontinuous (first order)</b> if two states coexist there. A <b>bifurcation</b> is a point where the system splits between two or more possible states.</div>")
        if not axes:
            tip = ("<div class='card'><h3>No transition can be resolved in this scan</h3><p>A transition needs at least four values of one numeric input, ideally 10 or more, so that its position can be pinned down. "
                   "Add for example <code>T = lin(1, 16, 12)</code> or <code>n_lef = log(2, 64, 10)</code> to the <code>[sweep]</code> section, with at least three replicates.</p></div>")
            return f'<section id="phases"><h2>Phase transitions and bifurcations</h2><p class="intro">Where does the system change its behaviour?</p>{defs}{tip}</section>'
        det = PH.detect(res)
        real = det[det.kind != "no significant change"].sort_values("strength", ascending=False)
        parts = []
        if real.empty:
            parts.append("<div class='note'>No significant change of any metric was found along the scanned inputs: the system stays in one phase in this range. Widen the range to look for a transition.</div>")
        else:
            real["grp"] = list(zip(real.axis, real.slice, real.lo.round(6), real.hi.round(6)))     # one row per transition, listing all metrics that show it
            rows = []
            for _, g in list(real.groupby("grp", sort=False))[:8]:
                r = g.iloc[0]
                names = ", ".join(A.label(m).split(" (")[0] for m in g.metric[:4]) + (f" (+{len(g) - 4})" if len(g) > 4 else "")
                rows.append([names, r.axis, r.slice or "-", r.kind, PH.classify(r)])
            parts.append('<div class="card"><h3>Detected transitions</h3>' + _table(["Order parameter", "Along", "Other inputs", "Kind", "Reading"], rows)
                         + "<p class='caption'>Ranked by how sharp, how significant and how many signatures agree (fluctuation peak, critical slowing down, split replicates). "
                           "Detection uses the grid you scanned: refine it around the transition to separate a steep crossover from a true transition.</p></div>")
            top = PH.best(res, real, 3)
            for r in top:
                axis, d = PH.slice_df(res, r)
                smp = PH.samples_by(res, d, r.metric, axis)
                title = f"{A.label(r.metric).split(' (')[0]} along {r.axis}"
                parts.append(_card(title, [P.phase_panel(res, r, d, axis)], EXPLAIN["phase"], r.slice, intro=PH.classify(r) + "."))
                figs = [P.bifurcation_diagram(res, r, d, axis, smp)] + ([P.distribution_at(res, r, smp, axis)] if len(smp) >= 3 else [])
                parts.append(_card(f"Branches and distributions: {A.label(r.metric).split(' (')[0]}", figs, EXPLAIN["bifurcation"] + " " + EXPLAIN["distribution"], r.slice))
            refine = [f"<li><code>{r.axis} = lin({r.lo:.3g}, {r.hi:.3g}, 12)</code> for {esc(A.label(r.metric).split(' (')[0])}</li>" for r in top[:3]]
            parts.append("<div class='note'><b>Zoom in:</b> to resolve the strongest transitions, scan a finer range with more replicates, for example<ul>" + "".join(refine) + "</ul></div>")
        if len(axes) >= 2 and res.design.kind == "grid" and not real.empty:
            r = real.iloc[0]
            other = next((a for a in axes if a.label != r.axis), None)
            if other is not None:
                d = real[(real.metric == r.metric) & (real.axis == r.axis)]
                parts.append(_card(f"Phase map: {A.label(r.metric).split(' (')[0]}", [P.phase_map(res, r.metric, r.axis, other.label, d)], EXPLAIN["phasemap"], f"{r.axis} against {other.label}"))
        size = next((a for a in res.design.axes if a.label == "n_beads" and df[a.label].nunique() >= 2), None)
        if size is not None and not real.empty and (len(axes) >= 1):
            r = real.iloc[0]
            ax = next(a for a in axes if a.label == r.axis)
            parts.append(_card(f"Finite-size scaling: {A.label(r.metric).split(' (')[0]}", [P.binder_plot(res, r.metric, ax, size)], EXPLAIN["binder"], "Binder cumulant", intro="Keep the CTCF density fixed with ctcf_spacing so that only the size changes."))
        return f'<section id="phases"><h2>Phase transitions and bifurcations</h2><p class="intro">Where does the system change its behaviour, and how?</p>{defs}{"".join(parts)}</section>'

    def quality(self):
        res, df = self.res, self.df
        c = A.convergence(res)
        parts = []
        if len(c):
            parts.append(_card("Convergence of the runs", [P.convergence(res)], EXPLAIN["convergence"], "mean loop length"))
            low, drift = (c.ess < 30).mean(), (c.drift > 0.10).mean()
            msg = (f"Median effective samples per run: {c.ess.median():.0f}. {low:.0%} of the runs have fewer than 30 and {drift:.0%} drift by more than 10% between the two halves. ")
            msg += ("Increase <code>n_steps</code> (more steps, not more snapshots, add independent information) or the number of replicates." if low > 0.2 or drift > 0.2 else "The runs look well equilibrated.")
            parts.insert(0, f'<div class="{"warn" if low > 0.2 or drift > 0.2 else "note"}">{msg}</div>')
        if res.failures:
            rows = [[p, r, e[:120]] for p, r, e in res.failures[:15]]
            parts.append(f'<div class="card"><h3>Failed runs</h3>{_table(["point", "replicate", "error"], rows)}</div>')
        return f'<section id="quality"><h2>Quality of the scan</h2><p class="intro">Can the numbers be trusted? Equilibration, effective sample size and failed runs.</p>{"".join(parts)}</section>'

    def setup(self):
        spec = self.spec
        sysrows = [[k, v] for k, v in spec.system.items()]
        base = [[k, v] for k, v in spec.base.items()]
        sketch = _img(P.system_sketch(spec), "polymer")
        return ('<section id="setup"><h2>Setup</h2><p class="intro">The synthetic polymer, the fixed parameters and the scan file that produced this report.</p>'
                f'<div class="card"><h3>The polymer</h3><figure>{sketch}</figure><p class="caption">Beads along the polymer and the CTCF sites. Orientation decides which leg of a ring a site can hold.</p></div>'
                f'<div class="grid2"><div class="card"><h3>System</h3>{_table(["Key", "Value"], sysrows)}</div><div class="card"><h3>Fixed parameters</h3>{_table(["Parameter", "Value"], base)}</div></div>'
                f'<div class="card"><h3>Scan file</h3><pre class="cmd">{esc(spec.source or "(built from Python)")}</pre></div>'
                + (f'<div class="card"><h3>Command</h3><pre class="cmd">{esc(spec.command)}</pre></div>' if spec.command else "") + '</section>')

    def glossary(self):
        items = [("Replicate", "One independent run with a different random seed; all parameter points use the same seeds, so differences between points are not random noise."),
                 ("Burn-in", "The first part of each run, discarded because the polymer is still forgetting its random start."),
                 ("Production mean", "The average of a metric over the snapshots after the burn-in: the number that is compared between points."),
                 ("Effective samples (ESS)", "Number of snapshots divided by the autocorrelation time: how many independent samples a run is really worth."),
                 ("Eta squared", "Share of the variance of a metric explained by one input alone (first-order effect)."),
                 ("Sobol S1 / ST", "Share of the variance explained by an input alone (S1) and together with all its interactions (ST)."),
                 ("Elasticity", "Percent change of a metric per percent change of an input."),
                 ("Parent / child loop", "A parent holds other loops and sits in none; a child sits inside another loop."),
                 ("Bound fraction", "Share of ring legs that sit within the bound radius of a CTCF site."),
                 ("Linking number (Lk)", "Integer that counts how often two closed loops wind around each other; 0 means unlinked."),
                 ("Leak", "A strand crossing that happens with the excluded volume on."),
                 ("Order / control parameter", "The metric that describes the state, and the input that is varied to look for a change of state."),
                 ("Susceptibility", "How strongly the order parameter fluctuates in time; it peaks at a transition."),
                 ("Critical slowing down", "Peak of the autocorrelation time at a continuous transition: the system relaxes very slowly."),
                 ("Bistability", "Two coexisting states at the same parameters (replicates end in different ones; confirm with more replicates); the sign of a first-order transition or bifurcation."),
                 ("Binder cumulant", "Shape measure U4 of the fluctuations, used to locate a continuous transition from the crossing of different system sizes."),
                 ("Design", "Which parameter points are simulated: a full grid, one input at a time, random (quasi-random) points, or Sobol points for variance-based indices.")]
        return '<section id="glossary"><h2>Glossary</h2><div class="card"><dl>' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items) + "</dl></div></section>"


def fmt_range(a) -> str:
    if a.is_range:
        return f"{'log' if a.log else 'lin'} {a.lo:g} to {a.hi:g}" + (f" ({a.n} points)" if a.n else "")
    return ", ".join(_g(v) for v in a.values)


def write_report(res, path) -> Path:
    """Write sensitivity_report.html and the combined tables (runs, points, importance) next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b = Builder(res)
    if not len(b.df):
        raise RuntimeError("no run finished: " + "; ".join(e for _, _, e in res.failures[:3]))
    body = [b.overview(), b.glance(), b.phases()]
    for gid, title, intro, names in GROUPS:
        body.append(b.group_section(gid, title, intro, names))
    body += [b.quality(), b.setup(), b.glossary()]
    nav = [("overview", "Overview"), ("glance", "At a glance"), ("phases", "Phase transitions")] + [(g[0], g[1]) for g in GROUPS] + [("quality", "Quality"), ("setup", "Setup"), ("glossary", "Glossary")]
    body_html = "".join(s for s in body if s)
    present = {m.group(1) for m in __import__("re").finditer(r'<section id="(\w+)"', body_html)}
    links = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in nav if i in present)
    chips = "".join(f'<span class="chip"><small>{esc(k)}</small>{esc(str(v))}</span>' for k, v in
                    (("design", res.spec.design), ("runs", len(res.tasks)), ("axes", ", ".join(a.label for a in res.design.axes)), ("beads", res.spec.system["n_beads"])))
    page = (HTML_DIR / "template.html").read_text(encoding="utf-8")
    fill = dict(title=esc(f"Sensitivity scan: {res.spec.name}"), css=(HTML_DIR / "style.css").read_text(encoding="utf-8") + EXTRA_CSS, nav=links, chips=chips,
                stamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), body=body_html)
    for key, val in fill.items():
        page = page.replace("{{%s}}" % key, val)
    path.write_text(page, encoding="utf-8")
    d = path.parent
    b.df.to_csv(d / "sensitivity_runs.csv", index=False)
    A.per_point(res).to_csv(d / "sensitivity_points.csv")
    b.imp.to_csv(d / "sensitivity_importance.csv", index=False)
    if b.sob is not None:
        b.sob.to_csv(d / "sensitivity_sobol.csv", index=False)
    return path
