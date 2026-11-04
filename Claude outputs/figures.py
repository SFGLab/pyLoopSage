"""Figure for the linking-number analysis."""
import numpy as np

from viz.theme import VIOLET, make_fig, save_figure

_EXPLAIN = (
    "Two loops are <em>linked</em> when one threads through the other, like two rings of a chain. "
    "The linking number (Gauss integral over the two closed curves) is an integer for closed loops; "
    "pairs with |linking number| above 0.5 count as linked. Left: how many such pairs each analysed "
    "frame contains. Right: how often each count occurs. Values near zero mean loops are well "
    "separated in space; many linked pairs would signal entangled, topologically constrained chromatin."
)


def links_figure(counts, stride):
    """Number of linked loop pairs per analysed MD frame, and its distribution."""
    fig, axs = make_fig(1, 2)
    axs[0].plot(np.arange(len(counts)) * stride, counts, color=VIOLET)
    axs[0].set(title="Linked loop pairs", xlabel="MD frame", ylabel="Number of linked pairs")
    axs[1].hist(counts, bins=np.arange(counts.min(), counts.max() + 2) - 0.5, color=VIOLET, alpha=0.75, edgecolor="white")
    axs[1].set(title="Distribution", xlabel="Number of linked pairs", ylabel="Frames")
    save_figure(fig, "links", title="Linked loop pairs in 3D", section="knots",
                caption=f"Linking numbers of disjoint loop pairs, every {stride}-th MD frame.", explain=_EXPLAIN)
