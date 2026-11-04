"""Figures of the Monte Carlo (loop extrusion) stage."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from analysis.loops import loop_lengths, occupancy
from analysis.mc_diagnostics import acf, detrend, integrated_time
from viz.panels import genomic_x, heatmap
from viz.theme import AQUA, INK, MUTED, ORANGE, VIOLET, make_fig, save_figure

_LEF_CMAP = LinearSegmentedColormap.from_list("lef", [VIOLET, "#b64fbf", ORANGE])


def _steps(res):
    return np.arange(res.n_samples) * res.meta["mc_step"]


def energy_traces(res):
    """Total energy, and its binding / crossing / folding components, vs MC step."""
    x, burn = _steps(res), res.burnin_idx * res.meta["mc_step"]
    fig, axs = make_fig(1, 2)
    axs[0].plot(x, res.Es, color=VIOLET)
    axs[0].set_title("Total energy")
    for k, (arr, name, c) in enumerate([(res.Bs, "Binding", ORANGE), (res.Ks, "Crossing", AQUA), (res.Fs, "Folding", VIOLET)]):
        axs[1].plot(x, arr, color=c, label=name)
    axs[1].set_title("Energy components")
    axs[1].legend()
    for ax in axs:
        ax.axvspan(0, burn, color="#ece8f5", alpha=0.8, lw=0)
        ax.text(burn / 2, ax.get_ylim()[1], "burn-in", ha="center", va="top", color=MUTED, fontsize=8)
        ax.set(xlabel="Monte Carlo step", ylabel="Energy (a.u.)")
    save_figure(fig, "energy_traces")


def convergence(res):
    """Autocorrelation of the folding energy and the unfolded fraction over time."""
    fig, axs = make_fig(1, 2)
    f = res.production(res.Fs)
    x = detrend(f) if res.meta["method"] == "Annealing" else f
    rho, tau = acf(x), integrated_time(x)
    lags = np.arange(len(rho)) * res.meta["mc_step"]
    axs[0].plot(lags, rho, color=VIOLET)
    axs[0].axhline(0, color=MUTED, lw=0.8)
    axs[0].axvline(tau * res.meta["mc_step"], color=ORANGE, ls="--", label=f"tau = {tau * res.meta['mc_step']:.0f} steps")
    axs[0].set(title="Autocorrelation of folding energy", xlabel="Lag (MC steps)", ylabel="Autocorrelation")
    axs[0].legend()
    axs[1].plot(_steps(res), res.ufs, color=AQUA)
    axs[1].set(title="Unfolded fraction of the chain", xlabel="Monte Carlo step", ylabel="Unfolded fraction (a.u.)")
    save_figure(fig, "convergence")


def lef_trajectories(res, geom, min_stable=3):
    """Each LEF as a coloured band between its two legs; rebinding jumps are masked."""
    jump = 0.4 * geom.n_beads
    fig, axs = make_fig(1, 1, cell=(8.5, 5.2))
    steps, n = _steps(res), res.Ms.shape[0]
    for i in range(n):
        m, nn = res.Ms[i].astype(float), res.Ns[i].astype(float)
        ok = np.ones(len(m), bool)
        ok[1:] = (np.abs(np.diff(m)) < jump) & (np.abs(np.diff(nn)) < jump)
        run = 0                                       # drop stable stretches shorter than min_stable
        for t in range(len(ok) + 1):
            if t < len(ok) and ok[t]:
                run += 1
                continue
            if 0 < run < min_stable:
                ok[t - run:t] = False
            run = 0
        to_mb = lambda b: (geom.start + b * geom.bp_per_bead) / 1e6
        axs[0].fill_between(steps, np.where(ok, to_mb(m), np.nan), np.where(ok, to_mb(nn), np.nan),
                            color=_LEF_CMAP(i / max(n - 1, 1)), alpha=0.55, lw=0)
    axs[0].set(xlabel="Monte Carlo step", ylabel=f"Position on {geom.chrom} (Mb)")
    axs[0].set_ylim(geom.start / 1e6, geom.end / 1e6)
    axs[0].set_title("LEF trajectories (band = extruded loop)")
    save_figure(fig, "lef_trajectories")


def occupancy_lengths(res, geom):
    """Where LEF legs sit, and how long the loops are (equilibrium)."""
    fig, axs = make_fig(1, 2)
    occ = occupancy(res, geom.n_beads)
    axs[0].fill_between(geom.axis_mb(), occ, color=VIOLET, alpha=0.4)
    axs[0].plot(geom.axis_mb(), occ, color=VIOLET, lw=1)
    axs[0].set(title="LEF leg occupancy", ylabel="LEF legs per bead (mean)")
    genomic_x(axs[0], geom)
    kb = geom.beads_to_bp(loop_lengths(res)) / 1e3
    axs[1].hist(kb, bins=40, color=VIOLET, alpha=0.75, edgecolor="white", density=True)
    axs[1].axvline(kb.mean(), color=ORANGE, ls="--", label=f"mean {kb.mean():.0f} kb")
    axs[1].set(title="Simulated loop lengths", xlabel="Loop length (kb)", ylabel="Probability density (1/kb)")
    axs[1].legend()
    save_figure(fig, "occupancy_lengths")


def loop_joint(res, geom):
    """Joint density of left and right leg positions."""
    fig, axs = make_fig(1, 1, cell=(5.8, 5))
    f = lambda b: (geom.start + b * geom.bp_per_bead) / 1e6
    hb = axs[0].hexbin(f(res.production(res.Ms).ravel()), f(res.production(res.Ns).ravel()),
                       gridsize=45, cmap="Purples", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=axs[0], shrink=0.82)
    cb.set_label("Snapshots x LEFs (log count)")
    axs[0].set(title="Where loops start and end", xlabel=f"Left leg on {geom.chrom} (Mb)", ylabel=f"Right leg on {geom.chrom} (Mb)")
    save_figure(fig, "loop_joint")


def mc_heatmap(mat, geom):
    fig, axs = make_fig(1, 1, cell=(6.2, 5.2))
    heatmap(fig, axs[0], mat, geom, "Loop-ensemble contact map", "Contact frequency (a.u., log)", log=True)
    save_figure(fig, "mc_heatmap")


def epi_trajectory(res, geom):
    fig, axs = make_fig(1, 1, cell=(8, 4.5))
    im = axs[0].imshow(res.epi, aspect="auto", origin="lower", cmap="Purples", interpolation="nearest",
                       extent=[0, res.n_samples * res.meta["mc_step"], geom.start / 1e6, geom.end / 1e6])
    axs[0].grid(False)
    axs[0].set(title="Epigenetic states over time", xlabel="Monte Carlo step", ylabel=f"Position on {geom.chrom} (Mb)")
    cb = fig.colorbar(im, ax=axs[0], shrink=0.82, ticks=range(int(res.epi.max()) + 1))
    cb.set_label("Potts state")
    save_figure(fig, "epi_trajectory")
