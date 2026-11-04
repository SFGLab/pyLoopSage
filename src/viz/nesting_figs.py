"""Nesting of loops: orphans (isolated), parents (hold loops, inside none) and children of order 1, 2, ... (inside k other loops)."""
import numpy as np

from analysis.nesting import kind_of, nesting, order_counts, pooled
from viz.theme import AQUA, INK, make_fig, MUTED, save_figure, VIOLET

PINK = "#e0457b"
COLORS = ["#1baf7a", INK, VIOLET, AQUA, PINK]
NAMES = ["orphans", "parents", "order 1", "order 2", "order 3+"]


def _legend(ax, ncol=2):
    """Legend below the axes so it never covers data."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, handlelength=1.4, columnspacing=1.2)


def _thirds(T):
    """Slices of the early, middle and late third of T snapshots."""
    e = np.linspace(0, T, 4).astype(int)
    return [slice(e[i], e[i + 1]) for i in range(3)]


def _by_time(parts, field, sl):
    """Valid values (and loop classes) of a field in a time slice, over all chains."""
    v = [getattr(p, field)[:, s][p.valid[:, s]] for p in parts for s in [sl(p.valid.shape[1])]]
    o = [kind_of(p.order, p.n_desc)[:, s][p.valid[:, s]] for p in parts for s in [sl(p.valid.shape[1])]]
    return np.concatenate(v), np.concatenate(o)


def nesting_figure(res, parts):
    """Loop counts per order over time, order distribution in time thirds, mean order over time."""
    c = res.chains[0]
    full = nesting(c.Ms, c.Ns)                         # chain 0 with burn-in
    t = np.arange(c.Ms.shape[1]) * c.meta["mc_step"]
    cnt = order_counts(full)
    fig, axs = make_fig(1, 3, cell=(5.0, 4.0))
    axs[0].stackplot(t, cnt, colors=COLORS, labels=NAMES, alpha=0.9, edgecolor="white", linewidth=0.3)
    axs[0].axvline(t[c.burnin_idx], color=MUTED, ls="--", lw=1.2)
    _legend(axs[0], 5)
    axs[0].set(title="Loops by nesting order", xlabel="MC step", ylabel="Number of loops", xlim=(t[0], t[-1]))
    w, labels = 0.27, ["early", "middle", "late"]
    for i, name in enumerate(labels):
        o = _by_time(parts, "order", lambda T, i=i: _thirds(T)[i])[1]
        frac = np.array([(o == k).mean() for k in range(5)])
        axs[1].bar(np.arange(5) + (i - 1) * w, frac, w, color=[VIOLET, "#b9a3f2", "#e6dcff"][i], edgecolor="white", label=f"{name} third")
    axs[1].set_xticks(range(5), ["orphan", "parent", "1", "2", "3+"])
    _legend(axs[1], 3)
    axs[1].set(title="Order distribution over time", xlabel="Loop class (children: nesting order k)", ylabel="Fraction of loops")
    tp = (np.arange(parts[0].valid.shape[1]) + c.burnin_idx) * c.meta["mc_step"]
    mean_o = np.mean([(p.order * p.valid).sum(0) / np.maximum(p.valid.sum(0), 1) for p in parts], axis=0)   # mean order per snapshot
    max_o = np.mean(np.stack([np.where(p.valid, p.order, 0).max(0) for p in parts]), axis=0)
    axs[2].plot(tp, max_o, color=PINK, lw=1.8, label="deepest order")
    axs[2].plot(tp, mean_o, color=VIOLET, lw=2.2, label="mean order")
    _legend(axs[2])
    axs[2].set(title="Nesting depth over time", xlabel="MC step", ylabel="Order", ylim=(0, None))
    save_figure(fig, "loop_nesting")


def parents_children_figure(res, parts, geom):
    """Parent length vs number of descendants, loop length per order, child/parent length ratio over time."""
    kb = geom.bp_per_bead / 1e3
    fig, axs = make_fig(1, 3, cell=(5.0, 4.0))
    nd, o = pooled(parts, "n_desc")
    ln, _ = pooled(parts, "length")
    ln *= kb
    par = (o == 0) & (nd > 0)                           # parents only: they hold loops
    kind = kind_of(o, nd)
    axs[0].scatter(ln[par], nd[par], s=8, color=VIOLET, alpha=0.25, edgecolor="none")
    if par.sum() > 20:                                  # binned mean descendants
        e = np.quantile(ln[par], np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(ln[par], e[1:-1]), 0, 7)
        keep = [b for b in range(8) if (idx == b).any()]                  # skip empty bins
        xm = np.array([ln[par][idx == b].mean() for b in keep])
        ym = np.array([nd[par][idx == b].mean() for b in keep])
        axs[0].plot(xm, ym, "-o", color=INK, lw=2, ms=5, label="binned mean")
        _legend(axs[0])
    axs[0].set(title="Parents and their descendants", xlabel="Parent loop length (kb)", ylabel="Loops inside the parent")
    lo, hi = max(ln.min(), 1e-3), ln.max()
    bins = np.logspace(np.log10(lo * 0.98), np.log10(hi * 1.02), 25)      # padded: safe for rounding and equal lengths
    for k in range(5):
        m = kind == k
        if m.sum() > 1:
            axs[1].hist(ln[m], bins=bins, density=True, histtype="step", lw=2, color=COLORS[k], label=f"{NAMES[k]} ({ln[m].mean():.0f} kb)")
    axs[1].set_xscale("log")
    _legend(axs[1], 2)
    axs[1].set(title="Loop length by order", xlabel="Loop length (kb)", ylabel="Density")
    for i, (name, col) in enumerate(zip(["early", "middle", "late"], ["#e6dcff", "#a98df0", VIOLET])):
        L, o2 = _by_time(parts, "length", lambda T, i=i: _thirds(T)[i])
        P, _ = _by_time(parts, "parent_len", lambda T, i=i: _thirds(T)[i])
        r = (L / P)[o2 >= 2]                                  # children only
        r = r[np.isfinite(r)]
        if len(r):
            axs[2].hist(r, bins=np.linspace(0, 1, 21), density=True, histtype="stepfilled", alpha=0.55, color=col, edgecolor=VIOLET, label=f"{name} (mean {r.mean():.2f})")
    _legend(axs[2], 3)
    axs[2].set(title="Child length / parent length", xlabel="Ratio to the smallest enclosing loop", ylabel="Density", xlim=(0, 1))
    save_figure(fig, "loop_parents_children")
