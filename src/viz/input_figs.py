"""Figures of the BEDPE anchors: map with orientations, statistics, loop configuration vs length.

- Figure of the compartment (A/B) input track.
- Figures of the preprocessed input data.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
from matplotlib.colors import ListedColormap, PowerNorm
from scipy.ndimage import gaussian_filter1d, maximum_filter
from scipy.stats import spearmanr

from analysis.inputs import CONF_MIN, CONFIGS
from reporting.explanations import EXPLAIN
from viz.theme import A_RED, AQUA, B_BLUE, FREE, INK, genomic_x, glow, heatmap, make_fig, make_mosaic, MUTED, ORANGE, save_figure, VIOLET


# ---- Figures of the BEDPE anchors: map with orientations, statistics, loop configuration vs length ----

CONFIG_COLOR = dict(convergent=VIOLET, tandem="#1baf7a", divergent=ORANGE, uncertain="#e0457b")
ORIENT_COLOR = dict(forward=VIOLET, reverse="#1baf7a", ambiguous="#e0457b", unknown="#7b7596")
_MARK = dict(forward=">", reverse="<", ambiguous="D", unknown="o")
_TXT = lambda k: dict(caption=EXPLAIN[k]["caption"], explain=EXPLAIN[k]["explain"], section="input", title=EXPLAIN[k]["title"])


def _mb(bp):
    return np.asarray(bp) / 1e6


def anchor_map(a, geom):
    """Loop arcs, anchor strengths with orientation, and orientation confidence along the region."""
    known = a.p_rev >= 0                                  # orientation panel only when orientations exist
    rows = ["arc", "str"] + (["conf"] if known.any() else [])
    fig, ax = make_mosaic([[r] for r in rows], size=(11.5, 9 if known.any() else 7.2), height_ratios=[2.2, 1.3, 0.8][:len(rows)], sharex=True)
    lo, hi = a.left_bp, a.right_bp
    w = a.loop_strength / (a.loop_strength.max() or 1)
    top = _mb(hi - lo).max()
    for i in np.argsort(-(hi - lo)):                      # long loops first, short ones on top
        ln = _mb(hi[i] - lo[i])
        ax["arc"].add_patch(Arc((_mb(lo[i] + hi[i]) / 2, 0), ln, 2 * ln, theta1=0, theta2=180, lw=0.8 + 2.4 * w[i],
                                color=CONFIG_COLOR[a.config[i]], alpha=0.8))
    ax["arc"].set(ylim=(0, top * 1.05), ylabel="Loop length (Mb)", title="Input loops (arc height = length, thickness = strength)")
    ax["arc"].legend(handles=[Line2D([], [], color=CONFIG_COLOR[c], lw=3, label=c) for c in CONFIGS],
                     loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8, ncol=4)
    x, o, conf = _mb(a.bp), a.orient, a.confidence
    for name in ("unknown", "ambiguous", "reverse", "forward"):
        m = o == name
        if not m.any():
            continue
        ax["str"].vlines(x[m], 0, a.strength[m], color=ORIENT_COLOR[name], lw=1.2, alpha=0.75)
        ax["str"].scatter(x[m], a.strength[m], marker=_MARK[name], s=26 + 40 * conf[m], color=ORIENT_COLOR[name],
                          edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
    ax["str"].set(ylabel="Anchor strength (loop weight)", title="Anchors: strength and CTCF orientation")
    ax["str"].legend(handles=[Line2D([], [], marker=_MARK[k], ls="", color=ORIENT_COLOR[k], label=k, markersize=7) for k in ORIENT_COLOR],
                     loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8, ncol=4)
    if not known.any():
        ax["str"].set(xlabel=f"Position on {geom.chrom} (Mb)", xlim=(geom.start / 1e6, geom.end / 1e6))
        save_figure(fig, "anchor_map", **_TXT("anchor_map"))
        return
    ax["conf"].bar(x[known], conf[known], width=_mb(geom.bp_per_bead) * 0.8, color=[ORIENT_COLOR[k] for k in o[known]])
    ax["conf"].axhline(CONF_MIN, color=MUTED, ls="--", lw=0.8)
    ax["conf"].set(ylim=(0, 1.05), ylabel="Orientation confidence", xlabel=f"Position on {geom.chrom} (Mb)",
                   title="How certain is the orientation call?  (dashed = uncertain below %.1f)" % CONF_MIN)
    ax["conf"].set_xlim(geom.start / 1e6, geom.end / 1e6)
    save_figure(fig, "anchor_map", **_TXT("anchor_map"))


def anchor_stats(a):
    """Strength by orientation, confidence histogram and loop-configuration counts."""
    fig, axs = make_fig(1, 3, cell=(4.8, 4.2))
    names = [n for n in ORIENT_COLOR if (a.orient == n).any()]
    data = [a.strength[a.orient == n] for n in names]
    bp = axs[0].boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color="#2b2540", lw=1.4))
    rng = np.random.default_rng(0)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color="#b9b0d4", lw=1)
    for k, (n, d) in enumerate(zip(names, data), 1):
        bp["boxes"][k - 1].set(facecolor=ORIENT_COLOR[n], alpha=0.35, edgecolor=ORIENT_COLOR[n])
        axs[0].scatter(k + rng.uniform(-0.18, 0.18, len(d)), d, s=12, color=ORIENT_COLOR[n], alpha=0.7, edgecolor="white", linewidth=0.3)
    axs[0].set_xticks(range(1, len(names) + 1), [f"{n}\n(n={len(d)})" for n, d in zip(names, data)])
    axs[0].set(title="Strength by orientation", ylabel="Anchor strength (loop weight)")
    conf = a.confidence[a.p_rev >= 0]
    axs[1].hist(conf, bins=np.linspace(0, 1, 21), color=VIOLET, alpha=0.75, edgecolor="white")
    axs[1].axvline(CONF_MIN, color=ORANGE, ls="--", label=f"uncertain below {CONF_MIN}")
    axs[1].set(title="Orientation confidence", xlabel="Confidence |2p - 1|  (0 = coin flip, 1 = certain)", ylabel="Anchors")
    axs[1].legend(fontsize=8)
    counts = [int((a.config == c).sum()) for c in CONFIGS]
    bars = axs[2].bar(CONFIGS, counts, color=[CONFIG_COLOR[c] for c in CONFIGS], alpha=0.85)
    for b, c in zip(bars, counts):
        axs[2].text(b.get_x() + b.get_width() / 2, c, f"{c / max(sum(counts), 1):.0%}", ha="center", va="bottom", fontsize=9, color=MUTED)
    axs[2].set(title="Loop configuration", ylabel="Loops")
    save_figure(fig, "anchor_stats", **_TXT("anchor_stats"))


def anchor_loops(a):
    """Loop strength vs length per configuration, and the convergent share by loop length."""
    fig, axs = make_fig(1, 2)
    kb = a.loop_len_bp / 1e3
    for c in CONFIGS:
        m = a.config == c
        axs[0].scatter(kb[m], a.loop_strength[m], s=34, color=CONFIG_COLOR[c], alpha=0.75, edgecolor="white", linewidth=0.5, label=f"{c} ({m.sum()})")
    axs[0].set(xscale="log", title="Loop strength vs length", xlabel="Loop length (kb)", ylabel="Loop strength (weight)")
    axs[0].legend(fontsize=8)
    edges = np.unique(np.quantile(kb, np.linspace(0, 1, 5)))
    if len(edges) > 2:
        idx = np.clip(np.digitize(kb, edges[1:-1]), 0, len(edges) - 2)
        mid = [f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(len(edges) - 1)]
        bottom = np.zeros(len(mid))
        for c in CONFIGS:
            share = np.array([np.mean(a.config[idx == i] == c) if (idx == i).any() else 0 for i in range(len(mid))])
            axs[1].bar(mid, share, bottom=bottom, color=CONFIG_COLOR[c], alpha=0.85, label=c)
            bottom += share
        axs[1].set(title="Configuration by loop length", xlabel="Loop length bin (kb, quartiles)", ylabel="Share of loops", ylim=(0, 1))
        axs[1].legend(fontsize=8, loc="lower right")
    save_figure(fig, "anchor_loops", **_TXT("anchor_loops"))


# ---- Figure of the compartment (A/B) input track ---------------------------

def compartment_figure(comp, anchor_signal, summ, geom):
    """Signed A/B track with domains and anchors, domain sizes, anchor strength per compartment."""
    fig, ax = make_mosaic([["track", "track"], ["size", "anch"]], size=(11, 7.4), height_ratios=[1.5, 1])
    x = geom.axis_mb()
    pos, neg = np.clip(comp, 0, None), np.clip(comp, None, 0)
    ax["track"].fill_between(x, pos, color=A_RED, alpha=0.4, step="mid")
    ax["track"].fill_between(x, neg, color=B_BLUE, alpha=0.4, step="mid")
    glow(ax["track"], x, gaussian_filter1d(comp, 0.8), "#4a3a8c", lw=1)
    ax["track"].axhline(0, color=MUTED, lw=0.8)
    for s, e, k in summ["domains"]:
        if s > 0:
            ax["track"].axvline(x[s] - geom.bp_per_bead / 2e6, color="#b9b0d4", lw=0.7, ls=":")
    if anchor_signal is not None and anchor_signal.max() > 0:       # anchor rug along the bottom
        sig = gaussian_filter1d(anchor_signal, 0.7)
        base = -1.05
        ax["track"].fill_between(x, base, base + 0.25 * sig / sig.max(), color="#2b2540", alpha=0.55, lw=0, label="input anchor strength")
        ax["track"].legend(loc="upper right", fontsize=8)
    ax["track"].set(ylim=(-1.1, 1.1), ylabel="Compartment signal (A > 0 red, B < 0 blue)",
                    title=f"Compartments: {summ['frac_a']:.0%} A, {1 - summ['frac_a']:.0%} B, {summ['n_domains']} domains")
    genomic_x(ax["track"], geom)
    top = max([summ["size_a"].max() if summ["size_a"].size else 0, summ["size_b"].max() if summ["size_b"].size else 0, 1])
    bins = np.linspace(0, top, 14)
    for d, c, n in ((summ["size_a"], A_RED, "A"), (summ["size_b"], B_BLUE, "B")):
        if d.size:
            ax["size"].hist(d, bins=bins, color=c, alpha=0.6, edgecolor="white", label=f"{n} (median {np.median(d):.0f} kb)")
    ax["size"].set(title="Domain sizes", xlabel="Domain length (kb)", ylabel="Domains")
    ax["size"].legend(fontsize=8)
    groups = [("A", summ["anchors_a"], A_RED), ("B", summ["anchors_b"], B_BLUE)]
    for i, (n, d, c) in enumerate(groups, 1):
        if d.size:
            ax["anch"].boxplot([d], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor=c, alpha=0.35, edgecolor=c), medianprops=dict(color="#2b2540"),
                               whiskerprops=dict(color="#b9b0d4"), capprops=dict(color="#b9b0d4"))
    ax["anch"].set_xticks([1, 2], [f"A ({len(summ['anchors_a'])} beads)", f"B ({len(summ['anchors_b'])} beads)"])
    p = summ["p"]
    ax["anch"].set(title="Anchor strength per compartment" + (f" (p = {p:.2g})" if np.isfinite(p) else ""),
                   ylabel="Input anchor strength per bead (a.u.)")
    save_figure(fig, "compartments", **{k: EXPLAIN["compartments"][k] for k in ("title", "caption", "explain", "section")})


# ---- Figures of the preprocessed input data --------------------------------

def binding_profiles(inputs, geom):
    """L/R anchor profiles along the region (+ loop-length histogram for loop files)."""
    lo, x = inputs.loops, geom.axis_mb()
    has_loops = len(lo.loop_lengths) > 0
    fig, axs = make_fig(1, 2 if has_loops else 1)
    axs[0].fill_between(x, lo.L, color=VIOLET, alpha=0.25)
    glow(axs[0], x, lo.L, VIOLET, lw=1, label="L (left anchor, up)")
    axs[0].fill_between(x, -lo.R, color=ORANGE, alpha=0.25)
    glow(axs[0], x, -lo.R, ORANGE, lw=1, label="R (right anchor, mirrored down)")
    axs[0].axhline(0, color="#b9b0d4", lw=0.8)
    axs[0].set_title("Anchor binding profiles")
    axs[0].set_ylabel("Signal (scaled, a.u.)")
    genomic_x(axs[0], geom)
    axs[0].legend()
    if has_loops:
        kb = geom.beads_to_bp(lo.loop_lengths) / 1e3
        axs[1].hist(kb, bins=25, color=VIOLET, alpha=0.75, edgecolor="white")
        axs[1].axvline(kb.mean(), color=ORANGE, ls="--", label=f"mean {kb.mean():.0f} kb")
        axs[1].set(title="Loop lengths in the input", xlabel="Loop length (kb)", ylabel="Number of loops")
        axs[1].legend()
    save_figure(fig, "binding_profiles")


def _dots(ax, mat, geom, title, label, cmap, gamma=0.5, fig=None, cbar=True):
    """Loop matrix as bold dots on a tinted page: entries dilated, colour on a gamma scale, chain backbone hidden."""
    m = np.asarray(mat, float).copy()
    n = len(m)
    i, j = np.indices(m.shape)
    m[np.abs(i - j) <= 1] = 0                             # chain neighbours are not loops
    m = maximum_filter(np.abs(m), size=max(5, n // 60 | 1))    # grow sparse bins into big visible blobs (display only)
    ax.set_facecolor("#f1eef9")
    vmax = float(np.percentile(m[m > 0], 99)) if (m > 0).any() else 1
    im = ax.imshow(np.ma.masked_equal(m, 0), cmap=cmap, norm=PowerNorm(gamma, vmin=0, vmax=vmax), extent=geom.extent_mb(),
                   aspect="equal", interpolation="nearest")
    ax.plot(*([[geom.start / 1e6, geom.end / 1e6]] * 2), color="#b9b0d4", lw=0.8)   # diagonal guide
    ax.grid(False)
    ax.set(title=title, xlabel=f"Position on {geom.chrom} (Mb)", ylabel=f"Position on {geom.chrom} (Mb)")
    if cbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label(label, color=INK)
        cb.outline.set_visible(False)


def loop_matrices(inputs, geom):
    """Loop adjacency J and weighted loop map J_loss, drawn as visible dots."""
    lo = inputs.loops
    fig, axs = make_fig(1, 2, cell=(5.6, 4.6))
    _dots(axs[0], lo.J, geom, "Loop adjacency J", "", ListedColormap([VIOLET]), fig=fig, cbar=False)
    _dots(axs[1], lo.J_loss, geom, "Weighted loop map J_loss", "Loop strength (a.u., gamma 0.5)", FREE, fig=fig)
    save_figure(fig, "loop_matrices")


def track_rho(v, inputs) -> float:
    """Spearman correlation of a track with the smoothed input anchor strength."""
    sig = gaussian_filter1d(inputs.loops.L + inputs.loops.R, 1.0)
    return float(spearmanr(gaussian_filter1d(v, 1.0), sig)[0]) if np.std(v) > 0 and sig.std() > 0 else float("nan")


def track_figures(inputs, geom):
    """Extra tracks (BigWig, LEF loading, compartments), at most 3 per figure."""
    tracks = [(n, v) for n, v in zip(inputs.bw_names, inputs.BWs)]
    if inputs.lef_track is not None:
        tracks.append(("LEF loading preference", inputs.lef_track))
    colors = [VIOLET, ORANGE, AQUA]
    for part, i in enumerate(range(0, len(tracks), 3), 1):
        chunk = tracks[i:i + 3]
        fig, axs = make_fig(len(chunk), 1, cell=(9.5, 2.4), sharex=True)
        for k, (ax, (name, v)) in enumerate(zip(axs, chunk)):
            ax.fill_between(geom.axis_mb(), v, color=colors[k], alpha=0.3)
            glow(ax, geom.axis_mb(), v, colors[k], lw=1)
            ax.set_title(f"{name}  (rho with anchors = {track_rho(v, inputs):.2f})")
            ax.set_ylabel("Signal (a.u.)")
        genomic_x(axs[-1], geom)
        save_figure(fig, f"tracks_{part}", title=f"Input tracks ({part})", section="input",
                    caption=EXPLAIN["tracks"]["caption"], explain=EXPLAIN["tracks"]["explain"])
