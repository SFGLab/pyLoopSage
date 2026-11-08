"""Results as arrays and tables: cohesin trajectories, loop lengths, energies, maps (from `run_mc`) and 3D data (from `run_md`)."""
import numpy as np
import pandas as pd

_UNITS = {"bp": 1.0, "kb": 1e3, "mb": 1e6}


def _chain(mc, chain):
    return mc.chains[chain]


def _cut(c, a, production):
    return c.production(a) if production else a


def _position(geom, beads, units):
    if units == "bead":
        return beads
    return (geom.start + np.asarray(beads) * geom.bp_per_bead) / _UNITS[units]


def _length(geom, beads, units):
    return beads if units == "bead" else np.asarray(beads) * geom.bp_per_bead / _UNITS[units]


# ---- Monte Carlo -------------------------------------------------------------

def steps(mc) -> np.ndarray:
    """Monte Carlo step of every snapshot."""
    return np.arange(mc.n_samples) * mc.meta["mc_step"]


def trajectories(mc, geom, units: str = "bead", production: bool = False, chain: int = 0):
    """Cohesin legs over time: (left, right), each an array (LEFs, snapshots). units: bead, bp, kb or mb (genomic position).
    `production=True` keeps the snapshots after the burn-in."""
    c = _chain(mc, chain)
    left, right = np.minimum(c.Ms, c.Ns), np.maximum(c.Ms, c.Ns)
    return tuple(_position(geom, _cut(c, a, production), units) for a in (left, right))


def loop_lengths(mc, geom, units: str = "kb", production: bool = False, chain: int = 0) -> np.ndarray:
    """Loop length of every LEF at every snapshot: array (LEFs, snapshots). units: bead, bp, kb or mb."""
    c = _chain(mc, chain)
    return _length(geom, _cut(c, np.abs(c.Ns - c.Ms), production), units)


def mean_loop_length(mc, geom, units: str = "kb", production: bool = False, chain: int = 0) -> np.ndarray:
    """Average loop length over the LEFs at every snapshot: array (snapshots,)."""
    return loop_lengths(mc, geom, units, production, chain).mean(axis=0)


def energies(mc, production: bool = False, chain: int = 0) -> pd.DataFrame:
    """Table, one row per snapshot: step, total, folding, binding and crossing energy, temperature, unfolded fraction."""
    c = _chain(mc, chain)
    first = c.burnin_idx if production else 0
    cut = lambda a: _cut(c, a, production)
    return pd.DataFrame({"step": steps(c)[first:], "total": cut(c.Es), "folding": cut(c.Fs), "binding": cut(c.Bs),
                         "crossing": cut(c.Ks), "temperature": cut(c.Ts), "unfolded_fraction": cut(c.ufs)})


def loops_table(mc, geom, snapshot: int = -1, chain: int = 0) -> pd.DataFrame:
    """The loops of one snapshot (default: the last): LEF number, anchors in beads and bp, length in kb."""
    c = _chain(mc, chain)
    left, right = np.minimum(c.Ms[:, snapshot], c.Ns[:, snapshot]), np.maximum(c.Ms[:, snapshot], c.Ns[:, snapshot])
    return pd.DataFrame({"lef": np.arange(len(left)), "left_bead": left, "right_bead": right,
                         "left_bp": geom.start + left * geom.bp_per_bead, "right_bp": geom.start + right * geom.bp_per_bead,
                         "length_kb": (right - left) * geom.bp_per_bead / 1e3})


def loop_coverage(mc, geom) -> np.ndarray:
    """Average number of loops spanning each bead (production snapshots of all chains): array (beads,)."""
    p = mc.pooled()
    ms, ns = p.production(np.minimum(p.Ms, p.Ns)), p.production(np.maximum(p.Ms, p.Ns))
    diff = np.zeros((geom.n_beads + 1, ms.shape[1]))
    for k in range(ms.shape[1]):
        np.add.at(diff[:, k], ms[:, k], 1)
        np.add.at(diff[:, k], ns[:, k], -1)
    return np.cumsum(diff, axis=0)[:-1].mean(axis=1)


def leg_density(mc, geom) -> np.ndarray:
    """Average number of LEF legs on each bead (production snapshots of all chains): array (beads,)."""
    p = mc.pooled()
    legs = np.concatenate([p.production(p.Ms), p.production(p.Ns)], axis=0)
    return np.bincount(legs.ravel(), minlength=geom.n_beads) / legs.shape[1]


def contact_map(mc, geom) -> np.ndarray:
    """Contact map implied by the loops: array (beads, beads)."""
    from analysis.maps import loop_map
    return loop_map(mc.pooled(), geom.n_beads)


# ---- Molecular dynamics --------------------------------------------------------

def coordinates(md, frame: int = None) -> np.ndarray:
    """3D positions in nm: all frames (frames, beads, 3), or one frame (beads, 3)."""
    return md.coords if frame is None else md.coords[frame]


def radius_of_gyration(md) -> np.ndarray:
    """Radius of gyration (nm) of every MD frame."""
    return md.rg


def distance(md, geom, a, b, units: str = "bead") -> np.ndarray:
    """Distance (nm) between two beads in every MD frame. a, b: bead numbers (units='bead') or genomic positions in bp (units='bp')."""
    if units == "bp":
        a, b = (int(geom.bead_of(x)) for x in (a, b))
    elif units != "bead":
        raise ValueError("units must be 'bead' or 'bp'")
    return np.linalg.norm(md.coords[:, a] - md.coords[:, b], axis=1)


def contact_map_md(md) -> np.ndarray:
    """Average contact map of the 3D structures: array (beads, beads)."""
    return md.avg_map
