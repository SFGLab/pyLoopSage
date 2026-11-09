"""Sandbox: build a small synthetic system by hand (single CTCF sites, loops, A/B compartments) and simulate it."""
import numpy as np

from loopsage.settings.loader import parse_region

_ORIENT = {"forward": 0.0, "reverse": 1.0, "unknown": -1.0}      # probability that the site is in the reverse orientation


class Sandbox:
    """A synthetic region. Add CTCF sites (or loops) and compartments, then read them with `read_inputs(cfg, geom, **box.arrays())`.

        box = Sandbox("chr1:0-3000000", n_beads=100)
        box.add_ctcf(500_000, "forward", strength=10).add_ctcf(1_500_000, "reverse", strength=8)
        box.add_compartment(1_500_000, 3_000_000, "B")
        cfg = box.config(n_steps=20000)
        inputs = ls.read_inputs(cfg, ls.geometry(cfg), **box.arrays())
    """

    def __init__(self, region: str = "chr1:0-3000000", n_beads: int = 100):
        r = parse_region(region)
        self.chrom, self.start, self.end = r["chrom"], int(r["region_start"]), int(r["region_end"])
        self.n_beads = int(n_beads)
        self._sites, self._loops = [], []
        self._comp = None

    @property
    def region(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"

    @property
    def bp_per_bead(self) -> int:
        return (self.end - self.start) // self.n_beads

    def _bead(self, bp) -> int:
        if not self.start <= bp < self.end:
            raise ValueError(f"position {bp} is outside the region {self.region}")
        return int((bp - self.start) // self.bp_per_bead)

    # ---- CTCF sites and loops -------------------------------------------
    def add_ctcf(self, position: int, orientation="forward", strength: float = 1.0):
        """One CTCF site. orientation: 'forward' (left anchor of a loop), 'reverse' (right anchor), 'unknown', or a probability of reverse in [0, 1]."""
        prob = _ORIENT[orientation] if isinstance(orientation, str) and orientation in _ORIENT else orientation
        if not isinstance(prob, (int, float)) or not (prob == -1 or 0 <= prob <= 1):
            raise ValueError("orientation must be 'forward', 'reverse', 'unknown' or a probability between 0 and 1")
        self._bead(position)
        self._sites.append((int(position), float(strength), float(prob)))
        return self

    def add_loop(self, left: int, right: int, weight: float = 1.0):
        """A fixed loop between two positions (instead of single CTCF sites)."""
        if self._bead(left) == self._bead(right):
            raise ValueError("the anchors of a loop must be on different beads")
        self._loops.append((int(left), int(right), float(weight)))
        return self

    def add_ctcfs(self, positions, orientation="forward", strength=1.0):
        """Several CTCF sites at once. `orientation` and `strength` are one value for all sites or one per site."""
        positions = list(np.atleast_1d(positions))
        per = lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v] * len(positions)
        orient, strength = per(orientation), per(strength)
        if len(orient) != len(positions) or len(strength) != len(positions):
            raise ValueError("orientation and strength need one value per position")
        for pos, o, w in zip(positions, orient, strength):
            self.add_ctcf(int(pos), o, w)
        return self

    @property
    def ctcf_sites(self):
        """The CTCF sites added so far as a table."""
        import pandas as pd
        return pd.DataFrame(self._sites, columns=["position", "strength", "reverse_probability"])

    # ---- compartments ----------------------------------------------------
    def _compartments(self) -> np.ndarray:
        if self._comp is None:
            self._comp = np.zeros(self.n_beads)
        return self._comp

    def add_compartment(self, start: int, end: int, state="A"):
        """Set the A/B signal on [start, end) bp. state: 'A' (+1), 'B' (-1) or a number in [-1, 1]."""
        value = {"A": 1.0, "B": -1.0}.get(state, state) if isinstance(state, str) else float(state)
        if not isinstance(value, float) or abs(value) > 1:
            raise ValueError("state must be 'A', 'B' or a number between -1 and 1")
        b0, b1 = self._bead(start), self._bead(min(end, self.end - 1))
        self._compartments()[b0:b1 + 1] = value
        return self

    def set_compartments(self, values):
        """Your own A/B signal: one value per bead (`n_beads` values), each between -1 (B) and 1 (A); other values raise ValueError."""
        from loopsage.preproc.arrays import check_compartments
        self._comp = check_compartments(values, self.n_beads)
        return self

    def random_compartments(self, mean_block_mb: float = 1.0, seed: int = None):
        """Alternating A and B blocks with random lengths (exponential, mean `mean_block_mb`)."""
        rng = np.random.default_rng(seed)
        comp, i, state = np.zeros(self.n_beads), 0, 1.0 if rng.random() < 0.5 else -1.0
        mean_beads = max(mean_block_mb * 1e6 / self.bp_per_bead, 1.0)
        while i < self.n_beads:
            n = max(int(rng.exponential(mean_beads)), 1)
            comp[i:i + n] = state
            i, state = i + n, -state
        self._comp = comp
        return self

    @property
    def compartments(self):
        """The A/B signal per bead (array) or None."""
        return None if self._comp is None else self._comp.copy()

    # ---- use with the functions -------------------------------------------
    def config(self, config=None, **parameters):
        """A Config for this region and resolution; keyword arguments are config keys (n_steps, seed, ...)."""
        from loopsage.api import config as make
        return make(config, region=self.region, **{"n_beads": self.n_beads, **parameters})

    def arrays(self) -> dict:
        """The sandbox as keyword arguments of `read_inputs`: loops (or sites) and compartments."""
        if self._sites and self._loops:
            raise ValueError("use either CTCF sites or loops in one sandbox, not both")
        if not self._sites and not self._loops:
            raise ValueError("add at least one CTCF site (add_ctcf) or loop (add_loop)")
        if self._loops:
            left, right, weight = zip(*self._loops)
            out = dict(loops=dict(left=left, right=right, weight=weight))
        else:
            pos, weight, prob = zip(*self._sites)
            out = dict(sites=dict(position=pos, weight=weight, reverse=prob))
        if self._comp is not None:
            out["compartments"] = self._comp
        return out
