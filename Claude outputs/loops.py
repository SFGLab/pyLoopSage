"""Loop statistics of the equilibrium ensemble."""
import numpy as np

_MOVES = ("Rebinding", "Sliding", "State flip")


def loop_lengths(res) -> np.ndarray:
    """All loop lengths (beads) of every LEF in every post-burn-in snapshot."""
    return (res.production(res.Ns) - res.production(res.Ms)).ravel()


def occupancy(res, n_beads: int) -> np.ndarray:
    """Average number of LEF legs sitting on each bead (anchor probability)."""
    legs = np.concatenate([res.production(res.Ms).ravel(), res.production(res.Ns).ravel()])
    return np.bincount(legs, minlength=n_beads) / res.production(res.Es).shape[0]


def acceptance_table(res) -> list:
    """[(move, proposed, accepted, rate)] for each move type that was used."""
    return [(name, int(p), int(a), a / p) for name, (p, a) in zip(_MOVES, res.acc) if p > 0]
