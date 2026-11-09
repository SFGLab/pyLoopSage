"""LEF velocities as statistical physics: distributions, diffusion, phase space and free-energy landscapes.

Every LEF is a particle with two legs. Between snapshots a leg moves by dx; velocity v = dx / dt
(dt in MC steps). Snapshots where a LEF was rebound (legs jump far) are never mixed with motion.

The loop length l = n - m is the main coordinate (loop midpoint is secondary). A LEF has one speed v >= 0:
its right leg moves at +v and its left leg at -v, so the loop opens at 2v. v is always an absolute value.

- P(v)          distribution of the speed v = |dl/dt| / 2, compared with a half-Gaussian (1D Maxwell-Boltzmann) of equal <v^2>
- MSD(tau)      mean squared change of loop length ~ tau^alpha: alpha = 1 diffusion, 2 ballistic extrusion, < 1 trapped
- C_v(tau)      length-velocity autocorrelation: < 0 = confined, > 0 = persistent motion
- F(l, v)       free-energy surface of loop length and speed
- F = -ln P     free-energy landscape in units of the effective kT
- speed         mean leg displacement per MC step (unsigned); v_mid / v_len = drift of midpoint / growth of length
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis, spearmanr

from loopsage.analysis.ensemble import anchor_profile


# ---- Trajectories without rebinding jumps -----------------------------------

def _chains(res, jump=None):
    """Per chain: legs (m, n), jump counter J (constant between rebindings), dt (MC steps), set speed per LEF."""
    for ch in res.chains:
        m, n = ch.production(ch.Ms).astype(float), ch.production(ch.Ns).astype(float)
        step = ch.meta["mc_step"]
        thr = jump or max(10, 4 * np.sqrt(step))               # a rebinding moves the legs far more than sliding does
        big = (np.abs(np.diff(m, axis=1)) + np.abs(np.diff(n, axis=1))) > thr
        J = np.concatenate([np.zeros((len(m), 1), int), np.cumsum(big, axis=1)], axis=1)
        yield m, n, J, float(step), np.asarray(ch.meta.get("lef_speed", np.ones(len(m))), float)


def strong_anchors(inputs) -> np.ndarray:
    """Beads above the 75th percentile of the non-zero input anchor signal."""
    sig = anchor_profile(inputs)
    nz = sig[sig > 0]
    return np.flatnonzero(sig >= (np.percentile(nz, 75) if nz.size else np.inf))


def _near(x, anchors):
    return np.abs(x[..., None] - anchors).min(axis=-1) if anchors.size else np.full(np.shape(x), np.nan)


def velocities(res, inputs, jump=None) -> dict:
    """Leg velocities (vm, vn), signed length velocity vl and speed v = |vl| / 2 between consecutive snapshots (no rebinding), with the state they started from."""
    keys = ("vm", "vn", "vl", "v", "mid", "length", "dist", "set")
    out = {k: [] for k in keys}
    anchors = strong_anchors(inputs)
    for m, n, J, dt, v0 in _chains(res, jump):
        ok = J[:, 1:] == J[:, :-1]
        vm, vn = np.diff(m, axis=1) / dt, np.diff(n, axis=1) / dt
        dist = np.minimum(_near(m[:, :-1], anchors), _near(n[:, :-1], anchors))
        for k, a in zip(keys, (vm, vn, vn - vm, np.abs(vn - vm) / 2, (m + n)[:, :-1] / 2, (n - m)[:, :-1], dist, np.broadcast_to(v0[:, None], vm.shape))):
            out[k].append(a[ok])
    return {k: np.concatenate(v) for k, v in out.items()}


# ---- Diffusion and velocity memory ------------------------------------------

def msd(res, n_lags: int = 12, jump=None) -> dict:
    """MSD of the loop length vs lag, per set-speed group when speeds were set (else one group).

    Returns lag (MC steps), msd (groups, lags) in beads^2, group labels and the exponent alpha of the pooled curve.
    """
    chains = list(_chains(res, jump))
    v_all = np.concatenate([c[4] for c in chains])
    het = np.ptp(v_all) > 1e-9
    u = np.unique(v_all)
    edges = (u[:-1] + u[1:]) / 2 if len(u) <= 3 else np.quantile(v_all, [1 / 3, 2 / 3])     # one group per speed level, else terciles
    edges = edges if het else []
    ng = len(edges) + 1
    n_t = min(c[0].shape[1] for c in chains)
    lags = np.unique(np.round(np.geomspace(1, max(n_t // 4, 2), n_lags)).astype(int))
    num, cnt = np.zeros((ng, len(lags))), np.zeros((ng, len(lags)))
    for m, n, J, dt, v0 in chains:
        c, grp = n - m, (np.digitize(v0, edges) if het else np.zeros(len(m), int))
        for a, k in enumerate(lags):
            ok = J[:, k:] == J[:, :-k]
            d2 = np.where(ok, (c[:, k:] - c[:, :-k]) ** 2, 0.0)
            for g in range(num.shape[0]):
                num[g, a] += d2[grp == g].sum()
                cnt[g, a] += ok[grp == g].sum()
    curves = num / np.maximum(cnt, 1)
    pooled = num.sum(axis=0) / np.maximum(cnt.sum(axis=0), 1)
    gv = np.array([v_all[np.digitize(v_all, edges) == g].mean() if (np.digitize(v_all, edges) == g).any() else np.nan for g in range(ng)]) if het else None
    keep = cnt.sum(axis=1) > 0
    t = lags * chains[0][3]
    good = pooled > 0
    alpha = float(np.polyfit(np.log(t[good][:5]), np.log(pooled[good][:5]), 1)[0]) if good.sum() > 2 else float("nan")
    labels = [f"set speed {x:.2f}" for x in gv[keep]] if het else ["all LEFs"]
    return dict(lag=t, msd=curves[keep], groups=labels, alpha=alpha)


def vacf(res, max_lag: int = 12, jump=None):
    """Autocorrelation of the length velocity vs lag in snapshots (1 at lag 0)."""
    max_lag = max(1, min(max_lag, min(c[0].shape[1] for c in _chains(res, jump)) - 3))
    num, cnt = np.zeros(max_lag + 1), np.zeros(max_lag + 1)
    for m, n, J, dt, _ in _chains(res, jump):
        v = np.diff(n - m, axis=1)
        for k in range(max_lag + 1):
            ok = J[:, k + 1:] == J[:, :v.shape[1] - k]
            num[k] += (v[:, :v.shape[1] - k] * v[:, k:] * ok).sum()
            cnt[k] += ok.sum()
    c = num / np.maximum(cnt, 1)
    return np.arange(max_lag + 1), c / c[0] if c[0] > 0 else c


# ---- Windows: phase-space points (position, velocity) of each LEF -----------

@dataclass
class LefWindows:
    lef: np.ndarray      # LEF index of each window
    chain: np.ndarray    # chain (replica) index
    mid: np.ndarray      # mean loop midpoint (beads)
    length: np.ndarray   # mean loop length (beads)
    speed: np.ndarray    # activity speed (beads / MC step)
    v_mid: np.ndarray    # midpoint drift (beads / MC step)
    v_len: np.ndarray    # length growth rate (beads / MC step)
    intrinsic: np.ndarray  # speed that was set for the LEF (1 = default)

    def __len__(self):
        return len(self.mid)


def _slope(x, dt):
    """Least-squares slope of each row of x (windows x time points) per unit of dt."""
    t = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2
    return (x * t).sum(axis=1) / (t ** 2).sum() / dt


def lef_windows(res, window: int = 8, jump=None) -> LefWindows:
    """Cut every LEF's trajectory into windows of `window` snapshots that contain no rebinding."""
    cols = {k: [] for k in ("lef", "chain", "mid", "length", "speed", "v_mid", "v_len", "intrinsic")}
    for ci, (m, n, J, dt, v0) in enumerate(_chains(res, jump)):
        for i in range(len(m)):
            for sid in np.unique(J[i]):
                idx = np.flatnonzero(J[i] == sid)
                k = len(idx) // window
                if k == 0:
                    continue
                mi, ni = m[i][idx[:k * window].reshape(k, window)], n[i][idx[:k * window].reshape(k, window)]
                vals = dict(lef=[i] * k, chain=[ci] * k, mid=(mi + ni).mean(axis=1) / 2, length=(ni - mi).mean(axis=1),
                            speed=(np.abs(np.diff(mi, axis=1)) + np.abs(np.diff(ni, axis=1))).mean(axis=1) / 2 / dt,
                            v_mid=_slope((mi + ni) / 2, dt), v_len=_slope(ni - mi, dt), intrinsic=[v0[i]] * k)
                for key, v in vals.items():
                    cols[key] += list(v)
    a = {k: np.array(v, float) for k, v in cols.items()}
    return LefWindows(a["lef"].astype(int), a["chain"].astype(int), *(a[k] for k in ("mid", "length", "speed", "v_mid", "v_len", "intrinsic")))


def binned(x, y, bins):
    """Mean and standard error of y in bins of x: (centres, mean, sem); sparse bins are NaN."""
    k = np.digitize(x, bins) - 1
    c = 0.5 * (bins[1:] + bins[:-1])
    cnt = np.array([(k == b).sum() for b in range(len(c))])
    mean = np.array([y[k == b].mean() if cnt[b] > 1 else np.nan for b in range(len(c))])
    sem = np.array([y[k == b].std() / np.sqrt(cnt[b]) if cnt[b] > 1 else np.nan for b in range(len(c))])
    return c, mean, sem


# ---- Free-energy landscapes and flow ----------------------------------------

def landscape_1d(res, n_beads: int, jump=None, sigma: float = 1.0) -> dict:
    """F(l) = -ln P(l) of the loop length (kT units, min = 0), the mean speed at l, and the net growth rate (drift) at l.

    For overdamped motion the drift follows the force, -dF/dl; rho is their Spearman correlation.
    """
    pos, vel = [], []
    for m, n, J, dt, _ in _chains(res, jump):
        ok = J[:, 1:] == J[:, :-1]
        pos.append((n - m)[:, :-1][ok])
        vel.append((np.diff(n - m, axis=1) / dt)[ok])
    pos, vel = np.concatenate(pos), np.concatenate(vel)
    speed = np.abs(vel) / 2
    b = np.clip(np.round(pos).astype(int), 0, n_beads - 1)
    cnt = np.bincount(b, minlength=n_beads)
    p = gaussian_filter1d(cnt.astype(float), sigma)
    drift = gaussian_filter1d(np.bincount(b, weights=vel, minlength=n_beads) / np.maximum(cnt, 1), sigma)
    F = -np.log(p / p.max() + 1e-6)
    force = -np.gradient(gaussian_filter1d(F, 1.5))
    seen = cnt > 0
    mean_speed = gaussian_filter1d(np.bincount(b, weights=speed, minlength=n_beads) / np.maximum(cnt, 1), sigma)
    return dict(x=np.arange(n_beads), F=F, drift=drift, force=force, seen=seen, speed=mean_speed,
                rho=float(spearmanr(drift[seen], force[seen])[0]) if seen.sum() > 3 else float("nan"))


def velocity_field(w: LefWindows, n_mid: int = 10, n_len: int = 6, min_count: int = 4):
    """Grid over (midpoint, length): centres, mean (v_mid, v_len) (NaN if sparse) and the number of windows (time spent)."""
    mb = np.linspace(w.mid.min(), w.mid.max() + 1e-9, n_mid + 1)
    lb = np.linspace(w.length.min(), w.length.max() + 1e-9, n_len + 1)
    i, j = np.digitize(w.mid, mb) - 1, np.digitize(w.length, lb) - 1
    vel, cnt = np.full((n_len, n_mid, 2), np.nan), np.zeros((n_len, n_mid))
    for a in range(n_mid):
        for b in range(n_len):
            s = (i == a) & (j == b)
            cnt[b, a] = s.sum()
            if s.sum() >= min_count:
                vel[b, a] = w.v_mid[s].mean(), w.v_len[s].mean()
    return 0.5 * (mb[1:] + mb[:-1]), 0.5 * (lb[1:] + lb[:-1]), vel, cnt


def speed_summary(w: LefWindows, v: dict, d: dict) -> dict:
    """Headline numbers: at-rest fraction, kinetic temperature, excess kurtosis, MSD exponent, speed-set vs measured exponent."""
    vleg, moving = v["v"], v["vl"][v["vl"] != 0]
    key = w.chain * 100000 + w.lef
    per_lef = np.array([w.speed[key == k].mean() for k in np.unique(key)])
    out = dict(n_windows=len(w), mean=float(w.speed.mean()), rest=float((vleg == 0).mean()), temp=float((vleg ** 2).mean()),
               kurt=float(kurtosis(moving)) if moving.size > 3 else np.nan, alpha=d["alpha"], exponent=np.nan,
               lef_cv=float(per_lef.std() / per_lef.mean()) if per_lef.mean() > 0 else np.nan,
               rho_length=float(spearmanr(w.length, w.speed)[0]) if len(w) > 2 else np.nan)
    if np.ptp(w.intrinsic) > 1e-9 and len(per_lef) > 2:               # speeds were set: how does the measured speed follow them?
        v0 = np.array([w.intrinsic[key == k][0] for k in np.unique(key)])
        ok = per_lef > 0
        out["exponent"] = float(np.polyfit(np.log(v0[ok]), np.log(per_lef[ok]), 1)[0]) if ok.sum() > 2 else np.nan
    return out


# ---- Free energy in more dimensions -----------------------------------------

def leg_landscape(res, n_beads: int, sigma: float = 1.0) -> np.ndarray:
    """F(m, n) = -ln P(m, n): free energy of the two leg positions (kT, min = 0); NaN where a pair is (almost) never visited."""
    P = np.zeros((n_beads, n_beads))
    for ch in res.chains:
        m, n = ch.production(ch.Ms).ravel(), ch.production(ch.Ns).ravel()
        np.add.at(P, (m, n), 1)
    P = P + P.T                                           # legs are unordered: fill both triangles
    P = gaussian_filter1d(gaussian_filter1d(P, sigma, axis=0), sigma, axis=1)
    P /= P.max()
    return np.where(P > 1e-4, -np.log(np.maximum(P, 1e-12)), np.nan)


def free_energy_surface(a, b, bins=(30, 22), sigma: float = 1.2, cap: float = 7.0):
    """F(a, b) = -ln P(a, b) on a smoothed grid (kT, min = 0, capped at `cap` where almost never visited): centres of a, of b, F[b, a]."""
    H, ea, eb = np.histogram2d(a, b, bins=bins)
    P = gaussian_filter1d(gaussian_filter1d(H, sigma, axis=0), sigma, axis=1)
    F = np.minimum(-np.log(np.maximum(P, 1e-9) / P.max()), cap)
    return 0.5 * (ea[1:] + ea[:-1]), 0.5 * (eb[1:] + eb[:-1]), F.T


def phase_free_energy(w: LefWindows, bins=(24, 18)):
    """F(l, v) = -ln P(l, v) of windows in the phase plane: edges of l, edges of v, F[v, l] (NaN where empty)."""
    H, le, ve = np.histogram2d(w.length, w.speed, bins=bins)
    return le, ve, np.where(H > 0, -np.log(np.maximum(H, 1) / H.max()), np.nan).T
