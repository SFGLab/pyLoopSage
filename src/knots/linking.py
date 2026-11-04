"""Gauss linking numbers between 3D chromatin loops (no external topology library needed).

- pyvista snapshots of linked loops (optional; skipped when pyvista is unavailable).
"""
import json
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from common.runtime import stage_bar
from logger import get_logger


# ---- Gauss linking numbers between 3D chromatin loops (no external topology library needed) ----

MIN_LOOP_BEADS = 4        # shorter stretches are not closed curves worth testing


def gauss_linking(a, b) -> float:
    """Linking number of two closed polygons a (n,3) and b (m,3)."""
    a1, b1 = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
    p1, p2 = a[:, None, :], a1[:, None, :]                      # segment of a (rows)
    p3, p4 = b[None, :, :], b1[None, :, :]                      # segment of b (columns)
    r13, r14, r23, r24 = p3 - p1, p4 - p1, p3 - p2, p4 - p2
    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    n1, n2 = unit(np.cross(r13, r14)), unit(np.cross(r14, r24))
    n3, n4 = unit(np.cross(r24, r23)), unit(np.cross(r23, r13))
    asin = lambda x, y: np.arcsin(np.clip((x * y).sum(-1), -1, 1))
    omega = asin(n1, n2) + asin(n2, n3) + asin(n3, n4) + asin(n4, n1)
    sign = np.sign((np.cross(p4 - p3, p2 - p1) * r13).sum(-1))
    return float((omega * sign).sum() / (4 * np.pi))


def _boxes_overlap(a, b) -> bool:
    """Bounding boxes of two closed curves intersect (otherwise they cannot be linked)."""
    return bool(np.all(a.min(0) <= b.max(0)) and np.all(b.min(0) <= a.max(0)))


def frame_links(coords, ms, ns):
    """One frame: linked rows (m1, n1, m2, n2, Lk), pair count, loop count, and (|midpoint gap|, closest approach) per pair."""
    lo, hi = np.minimum(ms, ns), np.maximum(ms, ns)
    loops = sorted({(int(m), int(n)) for m, n in zip(lo, hi) if n - m + 1 >= MIN_LOOP_BEADS})
    curves = [coords[m:n + 1] for m, n in loops]
    rows, near, n_pairs = [], [], 0
    for i, (m1, n1) in enumerate(loops):
        for j in range(i + 1, len(loops)):
            m2, n2 = loops[j]
            if not (n1 < m2 or n2 < m1):                         # nested / overlapping loops share chain: skip
                continue
            n_pairs += 1
            near.append(((m1 + n1 - m2 - n2) // 2, cdist(curves[i], curves[j]).min()))   # midpoint gap, closest approach
            if _boxes_overlap(curves[i], curves[j]):
                lk = gauss_linking(curves[i], curves[j])
                if abs(lk) >= 0.5:
                    rows.append((m1, n1, m2, n2, round(lk)))
    return np.array(rows, dtype=int).reshape(-1, 5), n_pairs, len(loops), np.array(near, dtype=float).reshape(-1, 2)


@dataclass
class LinkResult:
    frame: np.ndarray             # analysed MD frame of each linked pair
    pairs: np.ndarray             # (k, 5) m1, n1, m2, n2, Lk
    frames: np.ndarray            # analysed MD frame indices
    n_pairs: np.ndarray           # disjoint loop pairs tested per analysed frame
    n_loops: np.ndarray           # loops per analysed frame
    n_beads: int
    meta: dict = field(default_factory=dict)
    near: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # frame, |midpoint gap| (beads), closest approach (nm)
    bond: float = 1.0             # mean bond length (nm)

    @property
    def linked_per_frame(self) -> np.ndarray:
        return np.array([(self.frame == f).sum() for f in self.frames])

    def save(self, path) -> None:
        np.savez_compressed(path, frame=self.frame, pairs=self.pairs, frames=self.frames, n_pairs=self.n_pairs,
                            n_loops=self.n_loops, n_beads=self.n_beads, meta=np.array(json.dumps(self.meta)),
                            near=self.near, bond=self.bond)

    @classmethod
    def load(cls, path) -> "LinkResult":
        z = np.load(path, allow_pickle=False)
        return cls(z["frame"], z["pairs"], z["frames"], z["n_pairs"], z["n_loops"], int(z["n_beads"]), json.loads(str(z["meta"])), z["near"], float(z["bond"]))


def link_ensemble(md, mc, stride: int = 1) -> LinkResult:
    """Linking analysis of every `stride`-th MD frame."""
    frames = np.arange(0, len(md.frame_idx), max(stride, 1))
    fr, pairs, npairs, nloops, near = [], [], [], [], []
    with stage_bar("Linking numbers", len(frames)) as advance:
        for k in frames:
            i = md.frame_idx[k]
            rows, n_p, n_l, nr = frame_links(md.coords[k], mc.Ms[:, i], mc.Ns[:, i])
            fr += [k] * len(rows)
            pairs.append(rows)
            npairs.append(n_p)
            nloops.append(n_l)
            near.append(np.c_[np.full(len(nr), k), np.abs(nr[:, :1]), nr[:, 1:]])
            advance()
    return LinkResult(np.array(fr, dtype=int), np.concatenate(pairs) if pairs else np.zeros((0, 5), int), frames,
                      np.array(npairs), np.array(nloops), md.coords.shape[1], dict(stride=int(stride)),
                      np.concatenate(near) if near else np.zeros((0, 3)),
                      float(np.linalg.norm(np.diff(md.coords[0], axis=0), axis=1).mean()))


# ---- pyvista snapshots of linked loops (optional; skipped when pyvista is unavailable) ----

log = get_logger(__name__)
COLORS = ["#6a3de8", "#e8b23d", "#1baf7a", "#d1303f", "#2b62d9", "#a98df0"]


def render_still(pts, pairs, size=520):
    """RGB image of the chain (grey tube) with linked loops coloured."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    pts = np.asarray(pts, float) - np.mean(pts, axis=0)
    bond = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background("white")
    pl.add_mesh(pv.Spline(pts, 4 * len(pts)).tube(radius=0.22 * bond), color="#c9c6d3", smooth_shading=True)
    for k, (m1, n1, m2, n2, _) in enumerate(pairs):
        for j, (a, b) in enumerate(((m1, n1), (m2, n2))):
            seg = pts[a:b + 1]
            if len(seg) > 3:
                pl.add_mesh(pv.Spline(seg, 4 * len(seg)).tube(radius=0.42 * bond), color=COLORS[(2 * k + j) % len(COLORS)],
                            specular=0.8, smooth_shading=True)
    pl.camera_position = "iso"
    pl.camera.zoom(1.1)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def link_stills(coords, res):
    """[(title, image)] for the most and the least linked analysed frame; [] on any failure."""
    try:
        c = res.linked_per_frame
        out = []
        for title, k in (("Most links", int(np.argmax(c))), ("Fewest links", int(np.argmin(c)))):
            pairs = res.pairs[res.frame == res.frames[k]]
            out.append((f"{title}: {len(pairs)} linked pair(s)", render_still(coords[res.frames[k]], pairs)))
            if np.argmax(c) == np.argmin(c):
                break
        return out
    except Exception as e:                                          # noqa: BLE001 - optional extra
        log.warning("Could not render linked-loop snapshots: %s", e)
        return []
