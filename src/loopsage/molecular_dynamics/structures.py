"""Initial polymer shapes. One coordinate unit = 0.1 nm (the bond length).

- Minimal mmCIF / PSF writers (one bead = one residue, chain bonds as connections).
- Smooth 3D trajectories: align frames, then interpolate in time.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from loopsage.logger import get_logger


# ---- Initial polymer shapes. One coordinate unit = 0.1 nm (the bond length) ----

log = get_logger(__name__)


def _unit_vectors(n):
    v = np.random.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def random_walk(n, step=1.0):
    return np.vstack([np.zeros(3), np.cumsum(step * _unit_vectors(n - 1), axis=0)])


def confined_random_walk(n, box=5):
    """Lattice random walk clipped to a cube (folds the chain in)."""
    V = np.zeros((n, 3))
    for i in range(1, n):
        V[i] = np.clip(V[i - 1] + np.random.choice([-1, 1], size=3), -box, box)
    return V


def self_avoiding_random_walk(n, step=1.0, radius=0.5, eps=1e-3, tries=1000):
    """Random walk whose beads avoid overlap (best effort after `tries` attempts)."""
    pts = np.zeros((n, 3))
    for k in range(1, n):
        for _ in range(tries):
            cand = pts[k - 1] + step * _unit_vectors(1)[0]
            if np.linalg.norm(pts[:k] - cand, axis=1).min() >= 2 * radius - eps:
                break
        pts[k] = cand
    return pts


def circle(n, z_stretch=1.0):
    r = 1 / (2 * np.sin(np.pi / n))                  # unit spacing between neighbours
    a = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.cos(a), r * np.sin(a), z_stretch * np.arange(1, n + 1) / n])


def helix(n, radius=1, pitch=2):
    t = np.linspace(0, 4 * np.pi, n)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), np.linspace(0, pitch * n, n)])


def spiral(n, radius0=1, pitch=1, growth=0.05):
    t, r = np.linspace(0, 4 * np.pi, n), radius0 + growth * np.arange(n)
    return np.column_stack([r * np.cos(t), r * np.sin(t), np.linspace(0, pitch * n, n)])


def sphere(n, radius=1):
    phi, cos_t = np.random.uniform(0, 2 * np.pi, n), np.random.uniform(-1, 1, n)
    r, sin_t = radius * np.random.uniform(0, 1, n) ** (1 / 3), np.sqrt(1 - cos_t ** 2)
    return np.column_stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t])


SHAPES = {"rw": random_walk, "confined_rw": confined_random_walk, "self_avoiding_rw": self_avoiding_random_walk,
          "circle": circle, "helix": helix, "spiral": spiral, "sphere": sphere}


def make_structure(n: int, mode: str = "rw") -> np.ndarray:
    """(n, 3) starting coordinates for the chosen shape."""
    if mode not in SHAPES:
        raise ValueError(f"unknown initial structure '{mode}'; choose from {sorted(SHAPES)}")
    log.info("Initial structure: %s, %d beads", mode, n)
    return SHAPES[mode](n)


# ---- Minimal mmCIF / PSF writers (one bead = one residue, chain bonds as connections) ----

_ATOM_HEAD = """data_nucsim
#
_entry.id nucsim
#
_audit_conform.dict_name       mmcif_pdbx.dic
_audit_conform.dict_version    5.296
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic
# ----------- ATOMS ----------------
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""

_CONN_HEAD = """#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
"""


def write_mmcif(points, path) -> None:
    """Write bead coordinates (angstrom units, as the CIF format expects)."""
    n = len(points)
    atoms = "".join(f"ATOM {i + 1} D CA . ALA A 1 {i + 1} ? {x:.3f} {y:.3f} {z:.3f}\n"
                    for i, (x, y, z) in enumerate(np.asarray(points)))
    bonds = "".join(f"C{i + 1} covale ALA A {i + 1} CA ALA A {i + 2} CA\n" for i in range(n - 1))
    with open(path, "w") as f:
        f.write(_ATOM_HEAD + atoms + _CONN_HEAD + bonds)


def write_psf(n: int, path, title="LoopSage chain") -> None:
    """Topology for viewing the DCD trajectory in VMD/PyMOL."""
    lines = ["PSF CMAP\n", "\n", "      1 !NTITLE\n", f"REMARKS {title}\n", "\n", f"{n:>8} !NATOM\n"]
    lines += [f"{k:>8} BEAD {k:<5} ALA  CA   A      0.000000        1.00 0           0\n" for k in range(1, n + 1)]
    lines += ["\n", f"{n - 1:>8} !NBOND: bonds\n"] + [f"{i:>8}{i + 1:>8}\n" for i in range(1, n)]
    with open(path, "w") as f:
        f.writelines(lines)


# ---- Smooth 3D trajectories: align frames, then interpolate in time --------

def kabsch(p, q):
    """Rotate centred point set p onto q (least squares, no reflection)."""
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    return p @ (u @ np.diag([1, 1, d]) @ vt)


def align_frames(coords):
    """Centre every frame and rotate it onto the previous (already aligned) one."""
    out = coords - coords.mean(axis=1, keepdims=True)
    for k in range(1, len(out)):
        out[k] = kabsch(out[k], out[k - 1])
    return out


def interpolate_trajectory(coords, steps: int = 5, smooth: bool = True):
    """(frames, beads, 3) -> ((frames - 1) * steps + 1, beads, 3), aligned and smoothly interpolated."""
    x = align_frames(np.asarray(coords, dtype=float))
    if len(x) < 2 or steps < 2:
        return x
    t, tn = np.arange(len(x)), np.linspace(0, len(x) - 1, (len(x) - 1) * steps + 1)
    out = PchipInterpolator(t, x, axis=0)(tn)
    return gaussian_filter1d(out, 0.6 * steps, axis=0, mode="nearest") if smooth else out
