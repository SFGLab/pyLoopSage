"""Statistical-physics figures of LEF motion: set speeds, speed statistics, phase space and free-energy landscapes.

A LEF has one speed v >= 0: its right leg moves at +v and its left leg at -v (the loop opens at 2v).
"""
import numpy as np
from scipy.stats import lognorm

from analysis.speed import binned, free_energy_surface, landscape_1d, leg_landscape, msd, phase_free_energy, strong_anchors, vacf, velocity_field
from viz.theme import AQUA, INK, make_fig, MUTED, FREE, save_figure, VIOLET

PINK = "#e0457b"
TEAL = "#00a99d"              # overlay colour that does not occur in the free-energy colormap


def _kb(geom):
    return geom.bp_per_bead / 1e3


def _bar(ax, x, y, color, label=None, w=None):
    ax.bar(x, y, width=w, color=color, alpha=0.85, edgecolor="white", label=label)


# ---- Initial speeds ---------------------------------------------------------

def speed_init_figure(res):
    """Speeds given to the LEFs at the start (relative sliding rate, mean 1) and how they share the sliding moves."""
    m = res.meta
    v = np.concatenate([c.meta.get("lef_speed", np.ones(len(c.Ms))) for c in res.chains])
    kind = m.get("lef_speed_dist", "constant")
    fig, axs = make_fig(1, 2, cell=(5.6, 4.0))
    x = np.linspace(0, max(v.max() * 1.15, 1.5), 300)
    if kind == "constant":
        _bar(axs[0], [1.0], [1.0], VIOLET, w=0.08)
        axs[0].set_xlim(0, 2)
    else:
        axs[0].hist(v, bins=min(30, max(len(v) // 4, 6)), density=True, color=VIOLET, alpha=0.8, edgecolor="white", label="LEFs in this run")
        if kind == "maxwell":                           # speed of a 3D Gaussian velocity, scale a with mean speed 1
            a = np.sqrt(np.pi / 8)
            axs[0].plot(x, np.sqrt(2 / np.pi) * x ** 2 * np.exp(-x ** 2 / (2 * a ** 2)) / a ** 3, color=INK, lw=2, label="Maxwell-Boltzmann")
        elif kind == "lognormal":
            sg = m.get("lef_speed_spread", 0.5)
            axs[0].plot(x, lognorm.pdf(x, sg, scale=np.exp(-sg ** 2 / 2)), color=INK, lw=2, label="lognormal")
        else:
            axs[0].plot([], [], color=INK, label=f"{100 * m.get('lef_fast_fraction', 0.2):.0f}% fast, ratio {m.get('lef_fast_ratio', 5.0):g}")
        axs[0].legend(frameon=False, fontsize=8)
    axs[0].set(title=f"Initial LEF speeds ({kind.replace('_', ' ')})", xlabel="Relative speed (mean = 1)", ylabel="Density")
    s = np.sort(v)
    share = np.cumsum(s) / s.sum()                      # Lorenz curve: fraction of all sliding moves done by the slowest LEFs
    f = np.arange(1, len(s) + 1) / len(s)
    axs[1].plot([0, 1], [0, 1], "--", color=MUTED, lw=1.2, label="all equal")
    axs[1].plot(np.r_[0, f], np.r_[0, share], color=VIOLET, lw=2.4, label="this run")
    axs[1].set(title=f"Share of sliding moves (fastest 20% do {100 * (1 - np.interp(0.8, f, share)):.0f}%)", xlabel="Fraction of LEFs (slowest first)",
               ylabel="Fraction of sliding moves", xlim=(0, 1), ylim=(0, 1))
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    save_figure(fig, "lef_speed_init")


# ---- Speed statistics -------------------------------------------------------

def velocity_figure(res, v, summ, geom):
    """P(v) against a half-Gaussian (1D Maxwell-Boltzmann), MSD of the loop length, velocity autocorrelation."""
    kb, step = _kb(geom), res.meta["mc_step"]
    fig, axs = make_fig(1, 3, cell=(5.0, 4.2))
    dl = np.abs(np.round(v["vl"] * step).astype(int))                # |change of loop length| per snapshot (beads): a discrete lattice
    gap = max(int(np.gcd.reduce(dl[dl != 0])), 1) if (dl != 0).any() else 1
    kk = dl // gap
    k = np.arange(kk.max() + 1)
    p = np.bincount(kk, minlength=len(k)) / len(kk)
    g = np.exp(-(k * gap / 2) ** 2 / (2 * np.mean((dl / 2) ** 2)))
    x = k * gap / 2 / step * kb                                       # speed v = |dl| / (2 dt)
    axs[0].semilogy(x, np.where(p > 0, p, np.nan), "o-", color=VIOLET, ms=4, lw=1.5, label="simulation")
    axs[0].semilogy(x, g / g.sum(), "--", color=INK, lw=1.4, label="half-Gaussian, same <v$^2$>")
    axs[0].set(title=f"Speed P(v): {100 * summ['rest']:.0f}% at rest, <v> = {v['v'].mean() * kb:.1f}", xlabel="Speed v (kb / MC step)",
               ylabel="Probability", ylim=(max(p[p > 0].min() / 3, 1e-6), 1))
    axs[0].legend(frameon=False, fontsize=8)

    d = msd(res)
    cols = (AQUA, VIOLET, INK)[-len(d["groups"]):] if 1 < len(d["groups"]) <= 3 else [VIOLET] * len(d["groups"])
    for row, name, c in zip(d["msd"], d["groups"], cols):
        axs[1].loglog(d["lag"], row * kb ** 2, "o-", color=c, ms=4, lw=1.8, label=name)
    t0, y0, lo, hi = d["lag"][0], d["msd"].max(axis=0)[0] * kb ** 2, d["msd"].min() * kb ** 2, d["msd"].max() * kb ** 2
    for a, ls in ((1, ":"), (2, "-.")):
        axs[1].loglog(d["lag"][:6], y0 * (d["lag"][:6] / t0) ** a, ls, color=MUTED, lw=1.1, label=f"slope {a}")
    axs[1].set_ylim(lo / 3, hi * 30)
    axs[1].set(title=f"Loop-length MSD ~ t^a, a = {summ['alpha']:.2f}", xlabel="Lag t (MC steps)", ylabel="MSD (kb$^2$)")
    axs[1].legend(frameon=False, fontsize=8)

    lag, c = vacf(res)
    axs[2].plot(lag * step, c, "o-", color=VIOLET, ms=4, lw=1.8)
    axs[2].axhline(0, color=MUTED, lw=0.9)
    axs[2].set(title="Velocity autocorrelation", xlabel="Lag (MC steps)", ylabel="C(t) / C(0)")
    save_figure(fig, "lef_velocity")


# ---- Phase space and free energy --------------------------------------------

def _free_image(ax, fig, F, extent, label, aspect="auto"):
    """Free-energy map: dark = most probable (F = 0), yellow = rare; blank = never visited."""
    im = ax.imshow(F, origin="lower", extent=extent, aspect=aspect, cmap=FREE, interpolation="nearest")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(label)
    return im


def _surface(fig, ax, x, y, F, labels, title):
    """Free-energy surface (wells = probable states) with its contour map on the floor; replaces the 2D axes `ax`."""
    spec = ax.get_subplotspec()
    ax.remove()
    ax3 = fig.add_subplot(spec, projection="3d")
    X, Y = np.meshgrid(x, y)
    surf = ax3.plot_surface(X, Y, F, cmap=FREE, edgecolor="white", linewidth=0.15, antialiased=True, rstride=1, cstride=1, alpha=0.95)
    ax3.contourf(X, Y, F, zdir="z", offset=-0.85, levels=12, cmap=FREE, alpha=0.9)
    ax3.contour(X, Y, F, zdir="z", offset=-0.85, levels=12, colors="#111111", linewidths=0.5, alpha=0.5)       # contour lines of the shadow
    ax3.set(xlabel=labels[0], ylabel=labels[1], zlabel="F (kT)", title=title, zlim=(-0.85, F.max()))
    ax3.view_init(elev=32, azim=-58)
    fig.colorbar(surf, ax=ax3, shrink=0.6, pad=0.1).set_label("F = -ln P (kT)")
    return ax3


def phase_space_figure(w, v, res, inputs, geom):
    """F(length, speed) in the phase plane with the mean speed; F(length) as a line with the mean speed; kinetic temperature near anchors."""
    kb = _kb(geom)
    fig, axs = make_fig(1, 3, cell=(5.3, 4.4))
    le, ve, F = phase_free_energy(w)
    _free_image(axs[0], fig, F, [le[0] * kb, le[-1] * kb, ve[0] * kb, ve[-1] * kb], "Free energy F = -ln P (kT)")
    _trend_line(axs[0], w.length * kb, w.speed * kb, np.linspace(le[0] * kb, le[-1] * kb, 12), TEAL)
    axs[0].set(title="Phase space: free energy of (length, speed)", xlabel="Loop length (kb)", ylabel="Speed v (kb / MC step)")

    lc1 = landscape_1d(res, res.meta["n_beads"])
    x, seen = lc1["x"] * kb, lc1["seen"]
    axs[1].plot(x[seen], lc1["F"][seen], color=VIOLET, lw=2.2, label="F = -ln P")
    axs[1].set(title="Free energy of the loop length", xlabel="Loop length (kb)", ylabel="Free energy F (kT)")
    ax2 = axs[1].twinx()
    ax2._twin = True
    ax2.plot(x[seen], lc1["speed"][seen] * kb, color=INK, lw=1.6, label="mean speed")
    ax2.set_ylabel("Mean speed (kb / MC step)")
    ax2.set_ylim(bottom=0)
    ax2.grid(False)
    h1, l1 = axs[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axs[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center")

    d, t = v["dist"] * kb, v["v"] ** 2 * kb ** 2
    ok = ~np.isnan(d)
    c, m, e = binned(d[ok], t[ok] / t[ok].mean(), np.linspace(0, np.percentile(d[ok], 99), 10))
    axs[2].fill_between(c, m - e, m + e, color=VIOLET, alpha=0.25, lw=0)
    axs[2].plot(c, m, color=VIOLET, lw=2.2, marker="o", ms=4)
    axs[2].axhline(1, color=MUTED, ls="--", lw=1)
    axs[2].set(title="Kinetic temperature near anchors", xlabel="Distance of nearest leg to a strong anchor (kb)", ylabel="<v$^2$> / mean")
    save_figure(fig, "lef_phase_space")


def _trend_line(ax, x, y, bins, color):
    c, m, _ = binned(x, y, bins)
    ax.plot(c, m, color=color, lw=2.2, marker="o", ms=4)


def free_energy_figure(w, res, inputs, geom):
    """Free energy in more variables: F(midpoint, length) with the flow, and F(left leg, right leg)."""
    kb = _kb(geom)
    mb = lambda b: (geom.start + (np.asarray(b) + 0.5) * geom.bp_per_bead) / 1e6
    fig, axs = make_fig(1, 2, cell=(6.0, 4.6))
    cx, cy, vel, cnt = velocity_field(w)
    F = np.where(cnt > 0, -np.log(np.maximum(cnt, 1) / cnt.max()), np.nan)
    hx, hy = (cx[1] - cx[0]) / 2, (cy[1] - cy[0]) / 2
    _free_image(axs[0], fig, F, [mb(cx[0] - hx), mb(cx[-1] + hx), (cy[0] - hy) * kb, (cy[-1] + hy) * kb], "F = -ln P (kT)")
    X, Y = np.meshgrid(mb(cx), cy * kb)
    U, V = vel[..., 0] * kb / 1e3, vel[..., 1] * kb                  # signed drift of midpoint and length: the arrows' components
    ok = ~np.isnan(U)
    if ok.any():
        dx, dy = np.diff(mb(cx))[0], np.diff(cy * kb)[0]
        r = np.hypot(U / dx, V / dy)[ok]
        q = np.percentile(r, 90) or 1.0
        k = 0.9 / q * np.minimum(1, 1.3 / (r / q * 0.9 + 1e-12))     # most arrows fit a cell, outliers are clipped
        axs[0].quiver(X[ok], Y[ok], U[ok] * k, V[ok] * k, color=TEAL, angles="xy", scale_units="xy", scale=1, pivot="mid", width=0.007)
    axs[0].set(title="F(midpoint, length) and flow", xlabel=f"Loop midpoint on {geom.chrom} (Mb)", ylabel="Loop length (kb)")

    FL = leg_landscape(res, res.meta["n_beads"])
    _free_image(axs[1], fig, FL.T, geom.extent_mb(), "F = -ln P (kT)", aspect="equal")
    a = mb(strong_anchors(inputs))
    axs[1].plot(a, np.full_like(a, geom.extent_mb()[2]), "|", color=PINK, ms=8, label="strong anchors")
    axs[1].plot(np.full_like(a, geom.extent_mb()[0]), a, "_", color=PINK, ms=8)
    axs[1].set(title="F(left leg, right leg)", xlabel=f"Left leg on {geom.chrom} (Mb)", ylabel="Right leg (Mb)")
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    save_figure(fig, "lef_free_energy")


def surface_figure(w, geom):
    """Large stand-alone surface F(length, speed): static PNG/SVG on disk and an interactive viewer card in the report."""
    import base64
    import io

    import matplotlib.pyplot as plt
    from reporting.explanations import EXPLAIN
    from reporting.report import REGISTRY
    from viz.surface3d import surface_html
    kb = _kb(geom)
    x, y, F = free_energy_surface(w.length * kb, w.speed * kb, bins=(36, 26))
    fig, axs = make_fig(1, 1, cell=(9.0, 7.2))
    ax3 = _surface(fig, axs[0], x, y, F, ("Loop length (kb)", "Speed v (kb / MC step)"), "Free energy surface F(length, speed)")
    c, m, _ = binned(w.length * kb, w.speed * kb, np.linspace(x[0], x[-1], 14))
    ax3.plot(c, m, zs=-0.85, zdir="z", color=TEAL, lw=2.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    if REGISTRY.fig_dir is not None:
        (REGISTRY.fig_dir / "lef_free_energy_3d.png").write_bytes(buf.getvalue())
        fig.savefig(REGISTRY.fig_dir / "lef_free_energy_3d.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    text = EXPLAIN["lef_free_energy_3d"]
    img = base64.b64encode(buf.getvalue()).decode()
    html = (surface_html(x, y, F, ("Loop length (kb)", "Speed v (kb / MC step)")) +
            f'<noscript><img alt="{text["title"]}" src="data:image/png;base64,{img}" style="max-width:100%"></noscript>'
            f'<details open><summary>How to read this figure</summary><p>{text["explain"]}</p></details>')
    REGISTRY.add_block(text["title"], html, text["caption"], text["section"])
