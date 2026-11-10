"""Figures of the Monte Carlo (loop extrusion) stage.

- Epigenetic model (only when enabled): states over time, domains, link to loops.
- Figure comparing independent MC replicas (<= 3 panels).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from loopsage.analysis import ensemble
from loopsage.analysis.ensemble import acf, detrend, integrated_time, occupancy
from loopsage.common.utils import moving_average
from loopsage.reporting.explanations import EXPLAIN
from loopsage.viz.theme import (
    A_RED,
    AB,
    AQUA,
    B_BLUE,
    COOL,
    genomic_x,
    glow,
    HEAT,
    heatmap,
    INK,
    make_fig,
    make_mosaic,
    MUTED,
    ORANGE,
    save_figure,
    VIOLET,
)


# ---- Figures of the Monte Carlo (loop extrusion) stage ---------------------

_LEF_CMAP = LinearSegmentedColormap.from_list("lef", ["#5a2bd8", "#1f6fe0", "#08a874", "#f2a900", "#f0501e", "#e5307f"])


def _steps(res):
    return np.arange(res.n_samples) * res.meta["mc_step"]


def energy_traces(res):
    """Total energy with its plateau, and component fluctuations around their plateaus."""
    x, burn = _steps(res), res.burnin_idx * res.meta["mc_step"]
    fig, axs = make_fig(1, 2)
    plateau, sd = res.production(res.Es).mean(), res.production(res.Es).std()
    axs[0].axhspan(plateau - sd, plateau + sd, color=VIOLET, alpha=0.12, lw=0)
    axs[0].axhline(plateau, color=VIOLET, ls="--", lw=1)
    glow(axs[0], x, res.Es, VIOLET)
    lo, hi = np.percentile(res.Es[1:], [0], axis=0), np.percentile(res.Es[1:], 100)
    axs[0].set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    axs[0].set(title="Total energy and its plateau", ylabel="Energy (a.u.)")
    axs[0].text(0.97, 0.9, f"plateau {plateau:.5g} ± {sd:.2g}\nstart {res.Es[0]:.2g} (off scale)", transform=axs[0].transAxes,
                ha="right", va="top", color=MUTED, fontsize=9)
    for arr, name, c in ((res.Bs, "Binding", ORANGE), (res.Ks, "Crossing", AQUA), (res.Fs, "Folding", VIOLET)):
        glow(axs[1], x, arr - res.production(arr).mean(), c, lw=1.3, label=f"{name} (plateau {res.production(arr).mean():.4g})")
    dev = np.concatenate([np.ravel(a - res.production(a).mean()) for a in (res.Bs, res.Ks, res.Fs)])
    span = np.percentile(np.abs(dev), 97) * 1.6 or 1
    axs[1].set(title="Components relative to their plateau", ylabel="Energy - plateau (a.u.)", ylim=(-span, span))
    axs[1].legend(loc="upper right", fontsize=8)
    for ax in axs:
        ax.axvspan(0, burn, color="#efe9fb", alpha=0.9, lw=0)
        ax.set_xlabel("Monte Carlo step")
        ax.text(burn / 2, ax.get_ylim()[0], "burn-in", ha="center", va="bottom", color=MUTED, fontsize=8)
    save_figure(fig, "energy_traces")


def convergence(res):
    """Autocorrelation of the folding energy and the unfolded fraction over time."""
    fig, axs = make_fig(1, 2)
    f = res.production(res.Fs)
    x = detrend(f) if res.meta["method"] == "Annealing" else f
    rho, tau = acf(x), integrated_time(x)
    lags = np.arange(len(rho)) * res.meta["mc_step"]
    glow(axs[0], lags, rho, VIOLET, halo=False)
    axs[0].axhline(0, color=MUTED, lw=0.8)
    axs[0].axvline(tau * res.meta["mc_step"], color=ORANGE, ls="--", label=f"tau = {tau * res.meta['mc_step']:.0f} steps")
    axs[0].set(title="Autocorrelation of folding energy", xlabel="Lag (MC steps)", ylabel="Autocorrelation")
    axs[0].legend()
    glow(axs[1], _steps(res), res.ufs, VIOLET, halo=False)
    axs[1].set(title="Unfolded fraction of the chain", xlabel="Monte Carlo step", ylabel="Unfolded fraction (a.u.)")
    save_figure(fig, "convergence")


def _bin_time(mat, max_cols=500):
    """Average consecutive snapshots so the image has at most max_cols columns."""
    k = -(-mat.shape[1] // max_cols)
    n = mat.shape[1] // k * k
    return mat[:, :n].reshape(mat.shape[0], -1, k).mean(axis=2), k


def _kymograph_data(res, n_beads):
    """(coverage, legs): loops covering each bead and LEF legs on each bead, per snapshot."""
    t = np.arange(res.n_samples)[None, :].repeat(res.Ms.shape[0], 0)
    lo, hi = np.minimum(res.Ms, res.Ns), np.maximum(res.Ms, res.Ns)
    diff = np.zeros((n_beads + 1, res.n_samples))
    np.add.at(diff, (lo, t), 1)
    np.add.at(diff, (hi + 1, t), -1)
    legs = np.zeros((n_beads, res.n_samples))
    np.add.at(legs, (res.Ms, t), 1)
    np.add.at(legs, (res.Ns, t), 1)
    return np.cumsum(diff, axis=0)[:n_beads], legs


def kymograph(res, geom, inputs=None):
    """Smooth heat-map kymographs of loop coverage (violet) and LEF legs (teal), with the time average."""
    cov, legs = _kymograph_data(res, geom.n_beads)
    fig, ax = make_mosaic([["cov", "legs", "avg"]], size=(14, 5.6), width_ratios=[3, 3, 1], sharey=True)
    mc_step, ext = res.meta["mc_step"], None
    for key, mat, label, gamma, cmap, sy in (("cov", cov, "Loops covering the bead", 0.6, HEAT, 0.5),
                                              ("legs", legs, "LEF legs on the bead", 0.5, COOL, 0.9)):
        img, k = _bin_time(mat)
        img = gaussian_filter(img, (sy, 1.0 if img.shape[1] > 30 else 0.6))          # smooth in position and time
        ext = [0, res.n_samples * mc_step, geom.end / 1e6, geom.start / 1e6]
        im = ax[key].imshow(img, aspect="auto", cmap=cmap, norm=PowerNorm(gamma, vmin=0, vmax=max(img.max(), 1e-9)),
                            extent=ext, interpolation="bicubic")
        ax[key].axvline(res.burnin_idx * mc_step, color="#2b2540", ls="--", lw=0.9)
        ax[key].grid(False)
        ax[key].set(xlabel="Monte Carlo step", title=label)
        cb = fig.colorbar(im, ax=ax[key], shrink=0.8, pad=0.02, location="bottom")
        cb.locator = MaxNLocator(5)
        cb.update_ticks()
        cb.outline.set_visible(False)
    ax["cov"].set_ylabel(f"Position on {geom.chrom} (Mb)")
    ax["cov"].text(res.burnin_idx * mc_step, geom.end / 1e6, " burn-in", fontsize=8, va="top", color="#2b2540")
    avg, y = res.production(cov).mean(axis=1), geom.axis_mb()
    ax["avg"].fill_betweenx(y, avg, color=VIOLET, alpha=0.3)
    glow(ax["avg"], avg, y, VIOLET, lw=1)
    if inputs is not None:
        sig = gaussian_filter1d(ensemble.anchor_profile(inputs), 1.0)
        ax["avg"].fill_betweenx(y, sig / (sig.max() or 1) * (avg.max() or 1), color=ORANGE, alpha=0.3, label="input anchors")
        ax["avg"].legend(loc="lower right", fontsize=7)
    ax["avg"].set(title="Time average", xlabel="Mean loops covering")
    ax["avg"].set_ylim(geom.end / 1e6, geom.start / 1e6)
    save_figure(fig, "kymograph")


def _zoom_window(res, n_snap, start=None):
    """Snapshot range [a, b) inside the production phase (default: its middle, i.e. near equilibrium)."""
    n_snap = int(min(max(n_snap, 10), res.n_samples - res.burnin_idx))
    mid = res.burnin_idx + (res.n_samples - res.burnin_idx - n_snap) // 2
    a = int(np.clip(mid if start is None else start, 0, res.n_samples - n_snap))
    return a, a + n_snap


def kymograph_zoom(res, geom, n_snap=120, start=None):
    """Overview with the chosen window marked, plus un-binned zooms of loop coverage and LEF legs in that window."""
    cov, legs = _kymograph_data(res, geom.n_beads)
    a, b = _zoom_window(res, n_snap, start)
    mc_step, x0, x1 = res.meta["mc_step"], a * res.meta["mc_step"], b * res.meta["mc_step"]
    y0, y1 = geom.end / 1e6, geom.start / 1e6
    fig, ax = make_mosaic([["all", "cov", "legs"]], size=(15, 5.2), width_ratios=[1, 1.5, 1.5])
    img, _ = _bin_time(cov)
    ax["all"].imshow(gaussian_filter(img, (0.5, 1.0)), aspect="auto", cmap=HEAT, norm=PowerNorm(0.6, vmin=0, vmax=max(img.max(), 1e-9)),
                     extent=[0, res.n_samples * mc_step, y0, y1], interpolation="bicubic")
    ax["all"].axvspan(x0, x1, color="#111", alpha=0.18, lw=0)
    ax["all"].axvline(x0, color="#111", lw=1.2)
    ax["all"].axvline(x1, color="#111", lw=1.2)
    ax["all"].grid(False)
    ax["all"].set(title="Where the zoom is taken", xlabel="Monte Carlo step", ylabel=f"Position on {geom.chrom} (Mb)")
    for key, mat, label, gamma, cmap in (("cov", cov, "Zoom: loops covering the bead", 0.7, HEAT), ("legs", legs, "Zoom: LEF legs on the bead", 0.6, COOL)):
        w = mat[:, a:b]
        im = ax[key].imshow(w, aspect="auto", cmap=cmap, norm=PowerNorm(gamma, vmin=0, vmax=max(w.max(), 1e-9)), extent=[x0, x1, y0, y1],
                            interpolation="nearest" if key == "legs" else "bilinear")          # no time binning: single moves stay visible
        ax[key].grid(False)
        ax[key].set(title=label, xlabel="Monte Carlo step")
        cb = fig.colorbar(im, ax=ax[key], shrink=0.8, pad=0.02, location="bottom")
        cb.locator = MaxNLocator(5)
        cb.update_ticks()
        cb.outline.set_visible(False)
    save_figure(fig, "kymograph_zoom", caption=f"Snapshots {a}-{b} (MC steps {x0:.0f}-{x1:.0f}), without time averaging.")


def _segments(m, n, jump, min_len=3):
    """Index ranges of one LEF between rebinding jumps (leg moves > jump beads); shorter ranges are dropped."""
    cut = np.flatnonzero((np.abs(np.diff(m)) > jump) | (np.abs(np.diff(n)) > jump)) + 1
    edges = np.r_[0, cut, len(m)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= min_len]


def lef_trajectories(res, geom, max_lefs=120):
    """Area occupied by each LEF between its legs (colour = LEF index); each stretch between rebindings is smoothed on its own."""
    n = res.Ms.shape[0]
    ids = np.sort(np.random.default_rng(0).choice(n, min(n, max_lefs), replace=False))
    fig, axs = make_fig(1, 1, cell=(11, 5.6))
    ax, x, bp = axs[0], _steps(res), geom.bp_per_bead
    to_mb = lambda v: (geom.start + (v + 0.5) * bp) / 1e6
    jump, win = max(5.0, 0.06 * geom.n_beads), max(res.n_samples // 120, 1)   # larger leg moves count as relocations
    alpha = 0.4 if len(ids) < 40 else 0.55                                  # fewer LEFs overlap more per band
    for i in ids:
        m, nn = np.minimum(res.Ms[i], res.Ns[i]).astype(float), np.maximum(res.Ms[i], res.Ns[i]).astype(float)
        color = _LEF_CMAP(i / max(n - 1, 1))
        for a, b in _segments(m, nn, jump):
            w = min(win, b - a)
            lo, hi = moving_average(m[a:b], w), moving_average(nn[a:b], w)
            ax.fill_between(x[a:b], to_mb(lo), to_mb(hi), color=color, alpha=alpha, lw=0)
    ax.axvspan(0, res.burnin_idx * res.meta["mc_step"], color="#efe9fb", alpha=0.8, lw=0)
    ax.text(res.burnin_idx * res.meta["mc_step"] / 2, geom.end / 1e6, "burn-in", ha="center", va="top", color=MUTED, fontsize=8)
    ax.set(xlim=(0, x[-1]), ylim=(geom.start / 1e6, geom.end / 1e6), xlabel="Monte Carlo step", ylabel=f"Position on {geom.chrom} (Mb)",
           title=f"LEF trajectories ({len(ids)} of {n} LEFs; shaded = extruded loop)")
    sm = plt.cm.ScalarMappable(cmap=_LEF_CMAP, norm=plt.Normalize(0, n - 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("LEF index")
    cb.outline.set_visible(False)
    T = np.asarray(res.Ts, float)                                          # temperature strip under the x axis
    strip = ax.inset_axes([0, -0.095, 1, 0.03])
    strip.imshow(T[None, :], aspect="auto", cmap="Purples", extent=[0, x[-1], 0, 1], vmin=0, vmax=max(T.max(), 1e-9))
    strip.set(xlim=(0, x[-1]), yticks=[], xticks=[])
    for sp in strip.spines.values():
        sp.set_visible(False)
    strip.text(-0.008, 0.5, "T", transform=strip.transAxes, ha="right", va="center", color=MUTED, fontsize=9)
    strip.text(0.004, 0.5, f"{T[0]:.1f}", transform=strip.transAxes, ha="left", va="center", color="white" if T[0] > 0.5 * T.max() else MUTED, fontsize=8)
    strip.text(0.996, 0.5, f"{T[-1]:.1f}", transform=strip.transAxes, ha="right", va="center", color="white" if T[-1] > 0.5 * T.max() else MUTED, fontsize=8)
    ax.xaxis.labelpad = 24
    save_figure(fig, "lef_trajectories")


def _anchor_window(profile, anchors, occ, half=12):
    """Mean occupancy in a +-half window around anchors; band = 5-95% of the same number of random positions."""
    n = len(occ)
    pad = np.r_[np.zeros(half), occ, np.zeros(half)]
    take = lambda c: np.array([pad[i:i + 2 * half + 1] for i in c if half <= i < n - half])
    real = take(anchors).mean(axis=0)
    rng = np.random.default_rng(0)
    rand = np.array([take(rng.integers(half, n - half, len(anchors))).mean(axis=0) for _ in range(200)])
    return np.arange(-half, half + 1), real, np.percentile(rand, [5, 50, 95], axis=0)


def occupancy_lengths(res, geom, inputs):
    """Occupancy vs anchor strength per bead, enrichment around strong anchors, and loop-length agreement."""
    ins = ensemble
    occ = occupancy(res, geom.n_beads)
    sig = ins.anchor_profile(inputs)
    fig, axs = make_fig(1, 3, cell=(5.0, 4.4))
    # A: every bead = one point; colour = position along the region
    xs, ys = sig / (sig.max() or 1), occ / (occ.max() or 1)
    sc = axs[0].scatter(xs, ys, c=geom.axis_mb(), cmap=_LEF_CMAP, s=34, edgecolor="white", lw=0.5, alpha=0.9)
    if (sig > 0).sum() > 4:
        edges = np.unique(np.quantile(xs[xs > 0], np.linspace(0, 1, 6)))
        mid = [(xs[(xs >= lo) & (xs <= hi)].mean(), np.median(ys[(xs >= lo) & (xs <= hi)])) for lo, hi in zip(edges[:-1], edges[1:]) if ((xs >= lo) & (xs <= hi)).any()]
        axs[0].plot(*zip(*mid), color="#111", lw=2, marker="o", ms=5, label="median of anchor-strength bins")
        axs[0].legend(fontsize=8, loc="upper left")
    rho = ins.anchor_recovery(res, inputs)["spearman"]
    axs[0].set(title=f"Occupancy vs anchor strength, rho = {rho:.2f}" if np.isfinite(rho) else "Occupancy vs anchor strength",
               xlabel="Input anchor strength (0-1)", ylabel="LEF leg occupancy (0-1)")
    fig.colorbar(sc, ax=axs[0], shrink=0.8, label=f"Position on {geom.chrom} (Mb)")
    # B: average occupancy around strong anchors compared with random positions
    sm = gaussian_filter1d(sig, 1.0)
    strong = np.flatnonzero((sm > np.percentile(sm[sm > 0], 75)) & (sm >= np.r_[sm[:1], sm[:-1]]) & (sm >= np.r_[sm[1:], sm[-1:]])) if (sm > 0).any() else np.array([], int)
    if len(strong) >= 2:
        d, real, (lo, med, hi) = _anchor_window(sig, strong, occ)
        axs[1].fill_between(d, lo, hi, color="#b9b6c4", alpha=0.5, label="random positions (5-95%)")
        axs[1].plot(d, med, color="#6b6b78", lw=1.2, ls="--")
        glow(axs[1], d, real, VIOLET, lw=2.4, label=f"around {len(strong)} strong anchors")
        axs[1].axvline(0, color="#111", lw=0.8, ls=":")
        axs[1].legend(fontsize=8, loc="upper right")
    axs[1].set(title="Occupancy around anchors", xlabel="Distance from anchor (beads)", ylabel="Mean LEF legs per bead")
    # C: loop lengths
    inp, sim, ks, _ = ins.length_agreement(res, inputs, geom)
    bins = np.linspace(0, max(sim.max(), inp.max() if inp.size else 0), 30)
    axs[2].hist(sim, bins=bins, color=VIOLET, alpha=0.55, density=True, label=f"Simulated (median {np.median(sim):.0f} kb)")
    axs[2].hist(sim, bins=bins, histtype="step", color=VIOLET, density=True, lw=1.6)
    if inp.size:
        axs[2].hist(inp, bins=bins, histtype="step", color="#111", density=True, lw=1.8, label=f"Input (median {np.median(inp):.0f} kb)")
    axs[2].set(title="Loop lengths" + (f", KS = {ks:.2f}" if np.isfinite(ks) else ""), xlabel="Loop length (kb)", ylabel="Probability density (1/kb)")
    axs[2].legend(fontsize=8)
    save_figure(fig, "occupancy_lengths")


def _input_loops(inputs, geom):
    """(left, right) beads of the input loops (BEDPE only; empty otherwise)."""
    lo = getattr(inputs, "loops", None)
    if lo is None or lo.fmt != "bedpe":
        return np.zeros((0, 2), int)
    return np.argwhere(np.triu(lo.J, 2) > 0)


def loop_landscape(res, geom, inputs=None, max_arcs=250):
    """Top: arcs of the simulated loops over all snapshots (above; darker = present more often) and of the input loops (below), arc height = loop length.
    Bottom: loop length against loop centre over all snapshots; black dots are the input loops."""
    mb = lambda b: (geom.start + np.asarray(b) * geom.bp_per_bead) / 1e6
    ms, ns = res.production(res.Ms), res.production(res.Ns)
    inp = _input_loops(inputs, geom)
    fig, ax = make_mosaic([["arc"], ["len"]], size=(9.5, 8.2), height_ratios=[1.25, 1])
    t = np.linspace(0, 1, 40)

    def arcs(a, pairs, sign, color, lw, alpha):
        for l, r in pairs:
            x0, x1 = mb(l), mb(r)
            a.plot(x0 + (x1 - x0) * t, sign * (x1 - x0) * np.sin(np.pi * t), color=color, lw=lw, alpha=alpha)

    allp = np.stack([ms.ravel(), ns.ravel()], axis=1)
    allp = allp[allp[:, 1] > allp[:, 0]]
    pairs, cnt = np.unique(allp, axis=0, return_counts=True)             # each distinct loop and how often it occurs
    top = np.argsort(cnt)[::-1][:max_arcs]
    freq = cnt[top] / len(ms)                                           # fraction of snapshots containing the loop
    for i in np.argsort(freq):                                          # frequent loops drawn last, on top
        arcs(ax["arc"], pairs[top[i]][None], 1, VIOLET, 1.1, float(np.clip(0.15 + freq[i], 0.15, 0.9)))
    arcs(ax["arc"], inp[np.random.default_rng(0).permutation(len(inp))[:max_arcs]], -1, INK, 0.8, 0.45)
    ax["arc"].axhline(0, color=INK, lw=0.9)
    n_in = f"{len(inp)} input loops" if len(inp) else "no input loops"
    ax["arc"].set(xlim=(mb(0), mb(geom.n_beads)), title=f"Simulated loops over {len(ms)} snapshots (above; darker = more frequent) and {n_in} (below)", ylabel="Loop length (Mb, arc height)", xlabel=f"Position on {geom.chrom} (Mb)")
    ax["arc"].yaxis.set_major_formatter(lambda v, _: f"{abs(v):g}")
    # bottom: loop length along the region, as a median and bands over windows of loop centres (loop lengths are whole beads, so a cloud of points looks like rows)
    L, R = ms.ravel(), ns.ravel()
    ok = R > L
    mid, ln = mb((L[ok] + R[ok]) / 2), geom.beads_to_bp(R[ok] - L[ok]) / 1e3
    edges = np.linspace(mb(0), mb(geom.n_beads), max(int(geom.length_bp / 5e5), 8) + 1)
    k = np.digitize(mid, edges) - 1
    cx, q = [], []
    for b in range(len(edges) - 1):
        v = ln[k == b]
        if len(v) >= 30:
            cx.append((edges[b] + edges[b + 1]) / 2)
            q.append(np.percentile(v, [10, 25, 50, 75, 90]))
    cx, q = np.array(cx), np.array(q)
    a = ax["len"]
    if len(cx) > 3:
        xs = np.linspace(cx[0], cx[-1], 200)
        q = np.exp(np.array([np.interp(xs, cx, gaussian_filter1d(np.log(q[:, c]), 1.0, mode="nearest")) for c in range(5)]).T)          # smooth on the log scale
        a.fill_between(xs, q[:, 1], q[:, 3], color=VIOLET, alpha=0.25, lw=0, label="middle 50% of simulated loops")
        a.plot(xs, q[:, 2], color=VIOLET, lw=2.4, label="median of simulated loops")
    if len(inp):
        a.scatter(mb((inp[:, 0] + inp[:, 1]) / 2), geom.beads_to_bp(inp[:, 1] - inp[:, 0]) / 1e3, s=12, color=INK, alpha=0.6, zorder=3, label="input loops")
    a.set(yscale="log", title="Loop length along the region", xlabel=f"Loop centre on {geom.chrom} (Mb)", ylabel="Loop length (kb)", xlim=(mb(0), mb(geom.n_beads)))
    a.legend(frameon=False, fontsize=8, loc="upper right")
    save_figure(fig, "loop_landscape")


def mc_heatmap(mat, geom):
    fig, axs = make_fig(1, 1, cell=(6.2, 5.2))
    heatmap(fig, axs[0], mat, geom, "Loop-ensemble contact map", "Contact frequency (a.u., log)", log=True)
    save_figure(fig, "mc_heatmap")


def epi_trajectory(res, geom):
    fig, axs = make_fig(1, 1, cell=(8, 4.5))
    im = axs[0].imshow(res.epi, aspect="auto", origin="lower", cmap=AB, interpolation="nearest", vmin=0, vmax=max(int(res.epi.max()), 1),
                       extent=[0, res.n_samples * res.meta["mc_step"], geom.start / 1e6, geom.end / 1e6])
    axs[0].grid(False)
    axs[0].set(title="Epigenetic states over time", xlabel="Monte Carlo step", ylabel=f"Position on {geom.chrom} (Mb)")
    cb = fig.colorbar(im, ax=axs[0], shrink=0.82, ticks=range(int(res.epi.max()) + 1))
    cb.set_label("Epigenetic state (low = B-like, high = A-like)")
    save_figure(fig, "epi_trajectory")


def epigenetic_model(res, geom, steps, energy, comp, n_states):
    """Average epigenetic state along the chromosome, model energy over time, same-state map."""
    S = res.production(res.epi).astype(float)
    fig, axs = make_fig(1, 3, cell=(5.2, 4.4))
    x = (geom.start + (np.arange(S.shape[0]) + 0.5) * geom.bp_per_bead) / 1e6
    mean, sd = S.mean(axis=1), S.std(axis=1)
    axs[0].fill_between(x, mean - sd, mean + sd, color=AQUA, alpha=0.45, lw=0, label="spread over time")
    axs[0].axhspan(-0.2, (n_states - 1) / 2, color=B_BLUE, alpha=0.06, lw=0)
    axs[0].axhspan((n_states - 1) / 2, n_states - 0.8, color=A_RED, alpha=0.06, lw=0)
    glow(axs[0], x, mean, VIOLET, label="average epigenetic state")
    if comp is not None and np.ptp(comp) > 0:
        c = (comp - comp.min()) / np.ptp(comp) * (n_states - 1)
        axs[0].plot(x, c, color=ORANGE, lw=1.0, alpha=0.8, label="input compartment signal (rescaled)")
    axs[0].set(title="Average epigenetic state", ylabel="State (low = B-like, high = A-like)", ylim=(-0.2, n_states - 0.8))
    genomic_x(axs[0], geom)
    axs[0].legend(loc="upper center", ncol=1, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.17))
    glow(axs[1], steps, energy, VIOLET)
    axs[1].axvline(res.meta["mc_step"] * res.burnin_idx, color=MUTED, ls="--", lw=1)
    axs[1].set(title="Epigenetic-model energy", xlabel="Monte Carlo step", ylabel="Energy (a.u.)")
    heatmap(fig, axs[2], ensemble.same_state_map(res.production(res.epi), n_states), geom,
            "Chance two beads share a state", "P(same state)", vmax=1.0)
    save_figure(fig, "epigenetic_model")


def epigenetic_free_energy(res, geom, fe, n_states):
    """Free-energy landscape over (A-like share, domain order) and its split into energy and entropy along the A/B share."""
    fig, axs = make_fig(1, 2, cell=(6.2, 4.8))
    land, e, path = fe["landscape"], fe["edges"], fe["path"]
    c = (e[:-1] + e[1:]) / 2
    top = np.nanmax(land)
    cs = axs[0].contourf(c, c, land.T, levels=np.linspace(0, top, 14), cmap="viridis_r", alpha=0.95)
    axs[0].contour(c, c, land.T, levels=np.linspace(0, top, 14), colors="white", linewidths=0.4, alpha=0.6)
    k0 = fe["burnin"]
    axs[0].plot(path[:k0 + 1, 0], path[:k0 + 1, 1], color="#111", lw=0.8, alpha=0.55, label="burn-in path")
    axs[0].scatter(path[k0:, 0], path[k0:, 1], s=9, color="white", edgecolor="#111", linewidth=0.4, zorder=3, label="production snapshots")
    axs[0].scatter(*path[0], marker="D", s=36, color=ORANGE, edgecolor="#111", zorder=4, label="start")
    cb = fig.colorbar(cs, ax=axs[0], shrink=0.85)
    cb.set_label("Free energy F = -ln P (kT)")
    axs[0].set(title="Free-energy landscape", xlabel="Share of A-like beads  (0 = all B, 1 = all A)", ylabel="Domain order  (links joining equal states)", xlim=(0, 1), ylim=(0, 1))
    axs[0].legend(loc="lower right", fontsize=7, frameon=True, facecolor="white", framealpha=0.85)
    x = fe["centres"]
    for y, col, lab in ((fe["F"], VIOLET, "free energy  F"), (fe["U"], ORANGE, "energy  U"), (fe["TS"], AQUA, "entropy term  -TS")):
        axs[1].plot(x, y, color=col, lw=2.0, marker="o", ms=4, label=lab)
    axs[1].axhline(0, color=MUTED, lw=0.8)
    axs[1].set(title="Energy against entropy along the A/B share", xlabel="Share of A-like beads  (0 = all B, 1 = all A)", ylabel="Relative to the free-energy minimum (kT)", xlim=(0, 1))
    axs[1].legend(fontsize=8, frameon=False)
    save_figure(fig, "epigenetic_free_energy")


def epigenetic_maps(loops, board, combined, geom):
    """Loop map, epigenetic checkerboard and their product."""
    from loopsage.analysis.maps import minmax
    fig, axs = make_fig(1, 3, cell=(5.2, 4.4))
    heatmap(fig, axs[0], minmax(np.log1p(loops)), geom, "Loop contact map", "Normalised contact (0-1)", vmax=1, gamma=0.6)
    heatmap(fig, axs[1], board, geom, "Epigenetic checkerboard", "<s_i><s_j> (A-A, B-B = +, A-B = -)", diverging=True, vmax=1, cmap=AB)
    heatmap(fig, axs[2], minmax(np.log1p(combined)), geom, "Combined map", "Normalised contact (0-1)", vmax=1, gamma=0.6)
    save_figure(fig, "epigenetic_maps")


def epigenetic_features(profile, geom, f_loop, f_comb, ref=None):
    """Compartment signal, insulation and P(s) of the loop-only map next to the combined map."""
    x = geom.axis_mb()
    kb = f_loop.s * geom.bp_per_bead / 1e3
    fig, axs = make_fig(1, 3, cell=(5.2, 4.2))
    axs[0].fill_between(x, f_comb.pc1, 0, where=f_comb.pc1 >= 0, color=A_RED, alpha=0.75, lw=0)
    axs[0].fill_between(x, f_comb.pc1, 0, where=f_comb.pc1 < 0, color=B_BLUE, alpha=0.75, lw=0)
    axs[0].plot(x, profile / (np.abs(profile).max() or 1), color="#111", lw=1.3, label="average epigenetic state")
    axs[0].plot(x, f_loop.pc1, color=MUTED, lw=1.2, ls="--", label="loops only")
    axs[0].axhline(0, color=MUTED, lw=0.8)
    axs[0].set(title="Compartment signal (PC1) of combined map", ylabel="PC1 (scaled)")
    axs[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    genomic_x(axs[0], geom)
    for f, c, lab in ((f_loop, MUTED, "loops only"), (f_comb, VIOLET, "loops + epigenetic")):
        axs[1].plot(x, f.insulation, color=c, lw=2, label=lab)
        axs[2].loglog(kb, f.ps / f.ps[0], color=c, lw=2, label=lab)
    axs[1].axhline(0, color=MUTED, lw=0.8)
    axs[1].set(title="Insulation score", ylabel="log2 insulation")
    axs[1].legend(frameon=False, fontsize=8)
    genomic_x(axs[1], geom)
    axs[2].set(title="Diagonal decay P(s)", xlabel="Separation s (kb)", ylabel="Mean contact (relative to s = 1)")
    save_figure(fig, "epigenetic_features")


def epigenetic_loops(res, geom, n_states):
    """Do loops bring beads of the same epigenetic state together? Plus state populations over time."""
    ag = ensemble.loop_state_agreement(res, n_states)
    fig, axs = make_fig(1, 2, cell=(5.8, 4.2))
    kb = ag["sep"] * geom.bp_per_bead / 1e3
    axs[0].plot(kb, ag["baseline"], color=MUTED, lw=1.6, marker="o", ms=4, label="any two beads, same distance")
    glow(axs[0], kb, ag["loops"], VIOLET, label="the two ends of a loop")
    axs[0].set(title="Loop ends and epigenetic state", xlabel="Loop length (kb)", ylabel="P(same epigenetic state)", xscale="log")
    axs[0].legend(frameon=False, fontsize=8)
    t = np.arange(res.n_samples) * res.meta["mc_step"]
    for s, col in zip(range(n_states), AB(np.linspace(0, 1, n_states))):
        axs[1].plot(t, (res.epi == s).mean(axis=0), color=col, lw=1.8, label=f"state {s}" + (" (B-like)" if s == 0 else " (A-like)" if s == n_states - 1 else ""))
    axs[1].axvline(res.meta["mc_step"] * res.burnin_idx, color=MUTED, ls="--", lw=1)
    axs[1].set(title="Share of beads in each state", xlabel="Monte Carlo step", ylabel="Fraction of beads", ylim=(0, 1))
    axs[1].legend(frameon=False, fontsize=8, ncol=min(n_states, 4))
    save_figure(fig, "epigenetic_loops")


# ---- Figure comparing independent MC replicas (<= 3 panels) ----------------

_COLORS = ["#6a3de8", "#111111", "#1baf7a", "#a98df0", "#4b2bb8", "#e0457b", "#f58a3c", "#2b62d9"]


def replica_figure(res, geom, summ):
    """Energy traces, occupancy profiles and the replica-to-replica correlation."""
    n = len(summ["energy"])
    fig, axs = make_fig(1, 3, cell=(4.9, 4.2))
    step = res.meta["mc_step"]
    for k, e in enumerate(summ["energy"]):
        t = (np.arange(len(e)) + res.burnin_idx) * step
        axs[0].plot(t, moving_average(e, max(3, len(e) // 25)), color=_COLORS[k % 8], lw=1.6, label=f"replica {k}")
    axs[0].set(title=f"Energy after burn-in, R-hat = {summ['rhat']:.3f}", xlabel="Monte Carlo step", ylabel="Total energy (a.u.)")
    if n <= 6:
        axs[0].legend(fontsize=8)
    x = geom.axis_mb()
    occ = summ["occ"]
    axs[1].imshow(occ / (occ.max(axis=1, keepdims=True) + 1e-12), aspect="auto", cmap=HEAT, extent=[x[0], x[-1], n - 0.5, -0.5], interpolation="nearest")
    axs[1].grid(False)
    axs[1].set(title="Loop occupancy of each replica", xlabel=f"Position on {geom.chrom} (Mb)", ylabel="Replica", yticks=range(n))
    im = axs[2].imshow(summ["corr"], cmap=HEAT, vmin=min(0.0, np.nanmin(summ["corr"])), vmax=1)
    axs[2].set(title="Occupancy correlation", xlabel="Replica", ylabel="Replica", xticks=range(n), yticks=range(n))
    axs[2].grid(False)
    if n <= 8:
        for i in range(n):
            for j in range(n):
                axs[2].text(j, i, f"{summ['corr'][i, j]:.2f}", ha="center", va="center", fontsize=8, color="#111" if summ["corr"][i, j] < 0.75 else "#fff")
    fig.colorbar(im, ax=axs[2], shrink=0.8, label="Pearson r")
    save_figure(fig, "replicas", **dict(title=EXPLAIN["replicas"]["title"], caption=EXPLAIN["replicas"]["caption"],
                                        explain=EXPLAIN["replicas"]["explain"], section="monte_carlo"))
