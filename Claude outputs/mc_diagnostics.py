"""Convergence and equilibrium diagnostics of the Monte Carlo run."""
import numpy as np

TERMS = {"Es": "Total energy", "Bs": "Binding energy", "Ks": "Crossing energy",
         "Fs": "Folding energy", "ufs": "Unfolded fraction"}


def acf(x, max_lag=None) -> np.ndarray:
    """Normalised autocorrelation function (FFT based), lag 0 = 1."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    a = a / a[0] if a[0] > 0 else np.zeros(n)
    return a[: (max_lag or n // 2) + 1]


def integrated_time(x) -> float:
    """Integrated autocorrelation time in samples (Sokal self-consistent window)."""
    rho = acf(x, len(x) - 1)
    tau = 1.0
    for m in range(1, len(rho)):
        tau = 1 + 2 * rho[1:m + 1].sum()
        if m >= 5 * tau:
            break
    return float(max(tau, 1.0))


def detrend(x, deg: int = 3) -> np.ndarray:
    """Remove a polynomial trend (annealing drifts the energy)."""
    t = np.arange(len(x))
    return np.asarray(x) - np.polyval(np.polyfit(t, x, deg), t)


def equilibrium_summary(res) -> dict:
    """Post-burn-in mean/std per term, drift between halves, and effective sample size."""
    out, annealing = {}, res.meta["method"] == "Annealing"
    for key in TERMS:
        x = res.production(getattr(res, key))
        half = len(x) // 2
        drift = (x[half:].mean() - x[:half].mean()) / (x.std() + 1e-12) if half > 1 else 0.0
        out[key] = dict(mean=float(x.mean()), std=float(x.std()), drift=float(drift))
    x = res.production(res.Fs)
    tau = integrated_time(detrend(x) if annealing else x) if len(x) > 10 else 1.0
    out["tau"], out["ess"], out["n"] = tau, len(x) / tau, len(x)
    return out
