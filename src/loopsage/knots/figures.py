"""Figures for the optional linking analysis (<= 3 panels each)."""
import numpy as np
from scipy.ndimage import gaussian_filter

from loopsage.common.utils import moving_average
from loopsage.viz.theme import AQUA, MAGENTA, MUTED, ORANGE, HEAT, VIOLET, make_fig, save_figure

SEC = "knots"


def links_time_figure(res):
    """Linked pairs per frame, how often each count occurs, and the size of the linking numbers."""
    c, f = res.linked_per_frame, res.frames
    fig, axs = make_fig(1, 3)
    axs[0].plot(f, c, color=VIOLET, alpha=0.3, lw=1)
    axs[0].plot(f, moving_average(c.astype(float), max(3, len(c) // 20)), color=VIOLET, lw=2.2)
    axs[0].set(title="Linked pairs over time", xlabel="MD frame", ylabel="Linked loop pairs")
    axs[1].hist(c, bins=np.arange(c.min(), c.max() + 2) - 0.5, color=VIOLET, alpha=0.8, edgecolor="white")
    axs[1].axvline(c.mean(), color=ORANGE, ls="--", lw=1.4, label=f"mean {c.mean():.2f}")
    axs[1].set(title="Frames by number of links", xlabel="Linked pairs in a frame", ylabel="Frames")
    axs[1].legend()
    lk = np.abs(res.pairs[:, 4]) if len(res.pairs) else np.zeros(0, int)
    vals = np.arange(1, max(3, lk.max() if len(lk) else 1) + 1)
    axs[2].bar(vals, [(lk == v).sum() for v in vals], color=MAGENTA, width=0.6)
    axs[2].set(title="Size of linking number", xlabel="|Lk|", ylabel="Linked pairs (all frames)", xticks=vals)
    save_figure(fig, "links_time", title="Linked loops over the trajectory", section=SEC,
                caption=f"{len(f)} analysed frames; {int(c.sum())} linked pairs in total.",
                explain=_TIME)


def links_map_figure(res):
    """Where along the chain links occur: pair map, per-bead involvement, separation of linked loops."""
    n, nf = res.n_beads, max(len(res.frames), 1)
    p = res.pairs
    mid1, mid2 = (p[:, 0] + p[:, 1]) // 2, (p[:, 2] + p[:, 3]) // 2
    m = np.zeros((n, n))
    np.add.at(m, (mid1, mid2), 1)
    m = gaussian_filter(m + m.T, 1.5) / nf
    inv = np.zeros(n)
    for a, b, c, d in p[:, :4]:
        inv[a:b + 1] += 1
        inv[c:d + 1] += 1
    fig, axs = make_fig(1, 3)
    im = axs[0].imshow(m, cmap=HEAT, origin="lower", extent=[0, n, 0, n])
    axs[0].set(title="Which loops link", xlabel="Loop midpoint (bead)", ylabel="Loop midpoint (bead)")
    fig.colorbar(im, ax=axs[0], shrink=0.8, label="Links / frame")
    axs[1].fill_between(np.arange(n), inv / nf, color=VIOLET, alpha=0.25)
    axs[1].plot(np.arange(n), gaussian_filter(inv / nf, 1.5), color=VIOLET, lw=2)
    axs[1].set(title="Bead in a linked loop", xlabel="Bead", ylabel="Fraction of frames", xlim=(0, n))
    sep = np.abs(mid1 - mid2)
    axs[2].hist(sep, bins=np.linspace(0, n, 16), color=AQUA, alpha=0.9, edgecolor="white")
    axs[2].set(title="Distance between linked loops", xlabel="Midpoint separation (beads)", ylabel="Linked pairs")
    save_figure(fig, "links_map", title="Where loops link", section=SEC,
                caption="Position of linked loops along the chain, accumulated over all analysed frames.",
                explain=_MAP)


def links_approach_figure(res):
    """How close unlinked loops come to each other: closest-approach distribution, vs loop separation, and a risk curve."""
    near = res.near
    d, gap = near[:, 2] / res.bond, near[:, 1]
    fig, axs = make_fig(1, 3)
    axs[0].hist(d, bins=np.linspace(0, np.percentile(d, 99), 30), color=VIOLET, alpha=0.85, edgecolor="white")
    axs[0].axvline(1, color=ORANGE, ls="--", lw=1.4, label="1 bond length")
    axs[0].set(title="Closest approach of loop pairs", xlabel="Minimum distance (bond lengths)", ylabel="Loop pairs")
    axs[0].legend()
    edges = np.linspace(0, res.n_beads, 9)
    idx = np.clip(np.digitize(gap, edges) - 1, 0, 7)
    med = np.array([np.median(d[idx == i]) if (idx == i).any() else np.nan for i in range(8)])
    q1 = np.array([np.percentile(d[idx == i], 25) if (idx == i).any() else np.nan for i in range(8)])
    q3 = np.array([np.percentile(d[idx == i], 75) if (idx == i).any() else np.nan for i in range(8)])
    x = (edges[:-1] + edges[1:]) / 2
    axs[1].fill_between(x, q1, q3, color=VIOLET, alpha=0.2, label="25-75%")
    axs[1].plot(x, med, color=VIOLET, lw=2.2, marker="o", label="median")
    axs[1].set(title="Neighbours stay apart", xlabel="Gap between loop midpoints (beads)", ylabel="Closest approach (bond lengths)")
    axs[1].legend()
    cut = np.linspace(0.5, max(2, np.percentile(d, 60)), 40)
    frac = [(d < c).mean() for c in cut]
    axs[2].plot(cut, frac, color=MAGENTA, lw=2.4)
    axs[2].fill_between(cut, frac, color=MAGENTA, alpha=0.15)
    axs[2].set(title="Fraction of pairs that get close", xlabel="Distance threshold (bond lengths)", ylabel="Fraction of loop pairs")
    n_link = max(len(res.pairs), 0)
    save_figure(fig, "links_approach", title="How close do loops come to linking?", section=SEC,
                caption=f"{len(d)} loop-pair observations; {n_link} of them linked. Distances are minimum bead-to-bead distances.",
                explain=_APPROACH)


def links_stills_figure(stills):
    """3D snapshots (most / least linked frames) rendered by pyvista; stills = [(title, image)]."""
    fig, axs = make_fig(1, len(stills), cell=(5.0, 5.0))
    for ax, (title, img) in zip(axs, stills):
        ax.imshow(img)
        ax.set(title=title, xticks=[], yticks=[])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    save_figure(fig, "links_stills", title="Linked loops in 3D", section=SEC,
                caption="Grey: chromatin chain. Colours: loops that are linked, each pair in its own colour.",
                explain=_STILLS)


def links_examples_figure(stills):
    """Up to three examples of linked loops rendered by pyvista; stills = [(title, image)]."""
    fig, axs = make_fig(1, len(stills), cell=(5.0, 5.0))
    for ax, (title, img) in zip(axs, stills):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    save_figure(fig, "links_examples", title="Examples of linked loops in 3D", section=SEC,
                caption="Violet and black: the two linked loops (thin line: the chord that closes each). Gold marker: where they meet, the place of the link. Pale grey: the rest of the chain.",
                explain=_EXAMPLES)


def links_dynamics_figure(res):
    """How long links last and how many form and dissolve between analysed frames."""
    from loopsage.knots.linking import link_dynamics
    life, formed, dissolved = link_dynamics(res)
    fig, axs = make_fig(1, 2)
    top = int(life.max()) if len(life) else 1
    axs[0].hist(life, bins=np.arange(1, top + 2) - 0.5, color=VIOLET, alpha=0.85, edgecolor="white")
    axs[0].set(title="How long a link lasts", xlabel="Consecutive analysed frames", ylabel="Linked pairs", xticks=np.arange(1, top + 1) if top < 12 else None)
    axs[1].plot(res.frames, formed, color=VIOLET, lw=1.8, label="newly linked")
    axs[1].plot(res.frames, -dissolved, color=ORANGE, lw=1.8, label="no longer linked")
    axs[1].axhline(0, color=MUTED, lw=1)
    axs[1].set(title="Links forming and dissolving", xlabel="MD frame", ylabel="Pairs per analysed frame")
    axs[1].legend()
    save_figure(fig, "links_dynamics", title="Link lifetimes and turnover", section=SEC,
                caption=f"Median lifetime {np.median(life) if len(life) else 0:.0f} analysed frame(s); analysed frames are {res.meta.get('stride', 1)} MD frames apart.",
                explain=_DYNAMICS)


def crossings_figure(topo, barrier, strengths, curve, target, current):
    """Chain crossings per frame (leaks vs topoisomerase passages), the outcome of passage attempts, and the EV barrier curve."""
    fig, axs = make_fig(1, 3)
    f = np.arange(len(topo["leak"]))
    axs[0].bar(f, topo["leak"], color=ORANGE, label="leak (EV on)", width=1.0)
    axs[0].bar(f, topo["passage"], bottom=topo["leak"], color=VIOLET, label="topoisomerase passage", width=1.0)
    axs[0].set(title="Chain crossings per frame", xlabel="MD frame", ylabel="Crossings")
    axs[0].legend()
    names = ["attempts", "undone", "links\ncreated", "links\nremoved"]
    vals = [sum(topo["events"]), sum(topo["rejected"]), sum(topo["created"]), sum(topo["removed"])]
    axs[1].bar(names, vals, color=[MUTED, MAGENTA, ORANGE, VIOLET], width=0.6)
    axs[1].set_yscale("symlog")
    axs[1].set(title="Passage attempts" + ("" if topo.get("mode") == "local" else " (off: 'random' mode)"), ylabel="Count (all frames)")
    axs[2].plot(strengths, curve, color=VIOLET, lw=2.2)
    axs[2].axhline(5, color=MUTED, ls=":", lw=1)
    axs[2].axhline(target, color=ORANGE, ls="--", lw=1.4, label=f"target {target:g} kT")
    axs[2].scatter([current], [barrier], color="black", zorder=5, label=f"now: {barrier:.1f} kT")
    axs[2].set(title="Barrier against EV strength", xlabel="ev_ff_strength (kJ/mol)", ylabel="Crossing barrier (kT)", yscale="log")
    axs[2].legend()
    save_figure(fig, "crossings", title="Chain crossings and the excluded-volume barrier", section=SEC,
                caption=f"{sum(topo['leak'])} leaks and {sum(topo['passage'])} passages in {len(f)} frames; the crossing barrier is {barrier:.1f} kT.",
                explain=_CROSSINGS)


_TIME = ("Two loops are <em>linked</em> when one passes through the other like rings of a chain. Each loop is closed by "
         "a straight line between its two anchors, and the Gauss <em>linking number</em> Lk (an integer) counts how often the two "
         "rings wind around each other; pairs with |Lk| &ge; 1 are linked. Left: linked pairs in each analysed frame (line: "
         "smoothed). Middle: how often each count occurs; a peak at 0 means loops are usually free of each other. "
         "Right: |Lk| = 1 is a simple clasp, larger values are more tangled.")
_MAP = ("Left: each pixel is a pair of loops (axes: loop midpoints along the chain); darker means the pair links more often. "
        "Middle: fraction of frames in which a bead sits inside a linked loop; peaks mark regions that repeatedly get entangled. "
        "Right: how far apart (in beads) the linked loops are along the chain.")
_APPROACH = ("Even when no loops are linked, this shows how close they come. Left: the closest distance between two loops, in bond lengths "
             "(one bond = distance of neighbouring beads); linking needs one loop to pass through the other, so values well above 1 mean a safe margin. "
             "Middle: closest approach against how far apart the loops lie along the chain (line: median, band: middle half of the pairs); "
             "neighbouring loops are expected to be closer. Right: the share of loop pairs closer than a given distance; "
             "a steep rise at small distances indicates crowded loops at risk of entanglement.")
_STILLS = ("Snapshots of the frame with the most links (left) and one with few or none (right). The grey line is the whole chain; "
           "thick coloured stretches are loops that are linked with another loop, so you can see how they interlock.")

_EXAMPLES = ("Up to three different linked pairs, the largest |Lk| first. The two linked loops are drawn thick (violet and black); the thin line of the same colour is the "
           "chord that closes each loop, because linking is computed on the closed rings. The gold marker sits where the two loops come closest, i.e. at the place of the link. "
           "The rest of the chain is faded.")
_DYNAMICS = ("Left: a link is followed over consecutive analysed frames; the bars show for how many frames the same pair of loops stayed linked. Short lives mean links come and go; "
             "long lives mean they are trapped. Right: how many pairs became linked (up) or stopped being linked (down) compared with the previous analysed frame. Loops are also "
             "created and removed by the Monte Carlo, so a pair can disappear without the chains passing through each other.")
_CROSSINGS = ("<b>Crossings</b> are events in which two bonds of the chain pass through each other between two checks of the MD. If the excluded volume (EV) were a perfect wall there would be none. "
              "<b>Leaks</b> (orange) happen while the EV is on: the EV is too soft or the time step too large to keep strands apart. <b>Passages</b> (violet) involve a bead whose EV the topoisomerase "
              "switched off. Middle: passage attempts of the 'local' topoisomerase, how many were undone by the topology-simplifying rule (topo_keep), and how many links they created or removed. "
              "Right: the energy barrier for two bonds to cross, in units of kT (thermal energy), as a function of the EV strength; a barrier of a few kT is crossed easily, about 10 kT or more keeps the chain intact. "
              "The black dot is the current setting.")
