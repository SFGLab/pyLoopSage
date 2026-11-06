"""Topoisomerase as discrete strand-passage events (optional 'local' mode); helpers without the MD engine."""
import numpy as np
from scipy.spatial import cKDTree

from knots.linking import gauss_linking, MIN_LOOP_BEADS


def contact_sites(pos, radius: float, width: int, n_events: int, rng):
    """Random passage sites at close contacts: [(first, last+1, partner)], the beads first..last being the passing strand.

    A site is a pair of beads closer than `radius` and further apart than the strand along the chain; one of the two
    (chosen at random) becomes the centre of the passing strand of `width` beads.
    """
    pairs = cKDTree(pos).query_pairs(radius, output_type="ndarray")
    pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > width + 2]
    if not len(pairs) or n_events <= 0:
        return []
    out, n = [], len(pos)
    for q in rng.choice(len(pairs), min(n_events, len(pairs)), replace=False):
        a, b = pairs[q][rng.permutation(2)]
        g0 = int(np.clip(a - width // 2, 0, max(n - width, 0)))
        out.append((g0, min(g0 + width, n), int(b)))
    return out


def pair_links(pos, ms, ns, a: int, b: int) -> dict:
    """Linking numbers between the loops that contain bead a and those that contain bead b (disjoint loops only)."""
    lo, hi = np.minimum(ms, ns), np.maximum(ms, ns)
    loops = sorted({(int(m), int(n)) for m, n in zip(lo, hi) if n - m + 1 >= MIN_LOOP_BEADS})
    out = {}
    for x in (l for l in loops if l[0] <= a <= l[1]):
        for y in (l for l in loops if l[0] <= b <= l[1]):
            if x[1] < y[0] or y[1] < x[0]:
                out[x + y] = round(gauss_linking(pos[x[0]:x[1] + 1], pos[y[0]:y[1] + 1]))
    return out


def link_change(before: dict, after: dict):
    """(created, removed): pairs whose |Lk| grew or shrank between two evaluations."""
    keys = set(before) | set(after)
    grew = sum(abs(after.get(k, 0)) > abs(before.get(k, 0)) for k in keys)
    shrank = sum(abs(after.get(k, 0)) < abs(before.get(k, 0)) for k in keys)
    return grew, shrank


def keep_passage(created: int, keep: float, rng) -> bool:
    """Topology-simplifying rule: a passage that creates a link survives with probability `keep`; others always survive."""
    return created == 0 or keep >= 1.0 or bool(rng.random() < keep)
