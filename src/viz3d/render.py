"""Colourings of the 3D structure.

- Self-contained interactive 3D trajectory viewer (3d_viewer.html, no external libraries).
- pyvista rendering: shiny bead-chain videos (mp4) and poster images.
"""
import base64
import json
import warnings
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common.runtime import stage_bar
from logger import get_logger


# ---- Colourings of the 3D structure ----------------------------------------

PURPLE_CMAP = LinearSegmentedColormap.from_list("loopsage_depth", ["#3b1a9a", "#6f48e0", "#a98df0", "#e2d8ff"])   # near -> far
COMP_CMAP = LinearSegmentedColormap.from_list("loopsage_comp", ["#4a4fd6", "#a9a6ee", "#efeaf2", "#f19aa5", "#d1303f"])
MODES = ("purple", "rainbow", "compartments")
LABEL = dict(purple="Purple, depth cued (far = fainter)", rainbow="Rainbow by bead index",
             compartments="Compartments: B blue-purple, A red")


def scalars(mode, pts, cam_pos, radius, comp=None):
    """Per-bead scalar in [0, 1] (or [-1, 1] for compartments) and the matching cmap/clim."""
    n = len(pts)
    if mode == "purple":
        dist = np.linalg.norm(pts - np.asarray(cam_pos), axis=1)
        span = np.ptp(dist) or 1.0
        return ((dist - dist.min()) / span) ** 0.85, PURPLE_CMAP, (0, 1)        # per-frame range: strong depth cue
    if mode == "rainbow":
        return np.linspace(0, 1, n), "rainbow", (0, 1)
    if mode == "compartments":
        return np.asarray(comp, dtype=float), COMP_CMAP, (-1, 1)
    raise ValueError(f"unknown colouring '{mode}' (use {', '.join(MODES)})")


# ---- Self-contained interactive 3D trajectory viewer (3d_viewer.html, no external libraries) ----

log = get_logger(__name__)
TEMPLATE = Path(__file__).resolve().parents[1] / "html" / "viewer_template.html"


def build_viewer(frames, path, title="LoopSage 3D viewer", comp=None, fps: int = 20) -> Path:
    """Embed the trajectory (int16-quantised) and write the HTML file."""
    frames = np.asarray(frames, dtype=float)
    scale = float(np.abs(frames).max()) / 32000 or 1.0
    q = np.round(frames / scale).astype("<i2")
    meta = dict(frames=len(frames), beads=frames.shape[1], scale=scale, fps=fps,
                comp=None if comp is None else [round(float(v), 3) for v in comp],
                data=base64.b64encode(q.tobytes()).decode())
    html = TEMPLATE.read_text(encoding="utf-8").replace("{{title}}", title).replace("{{meta}}", json.dumps(meta))
    Path(path).write_text(html, encoding="utf-8")
    log.info("3D viewer: %d frames x %d beads, %.0f kB", len(frames), frames.shape[1], Path(path).stat().st_size / 1e3)
    return Path(path)


# ---- pyvista rendering: shiny bead-chain videos (mp4) and poster images ----


def _pyvista():
    """Import pyvista lazily; start a virtual display on headless Linux if needed."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    return pv


def _meshes(pv, pts, s, bond, representation="tube"):
    """Smooth spline tube (default) with optional sphere glyphs ("beads") carrying the colour scalars."""
    n = len(pts)
    spline = pv.Spline(pts, 4 * n)
    spline["s"] = np.interp(np.linspace(0, n - 1, spline.n_points), np.arange(n), s)
    out = {"tube": spline.tube(radius=0.4 * bond, n_sides=16)}
    if representation == "beads":
        spheres = pv.PolyData(pts)
        spheres["s"] = s
        out["beads"] = spheres.glyph(geom=pv.Sphere(radius=0.62 * bond, theta_resolution=14, phi_resolution=14), scale=False, orient=False)
        out["tube"] = spline.tube(radius=0.2 * bond, n_sides=10)       # thin connecting string
    return out


def render_video(frames, path, mode, comp=None, fps=30, size=640, poster=None, turns=1.0, representation="tube") -> Path:
    """Write an mp4 of the (already interpolated) trajectory while the camera orbits once."""
    pv = _pyvista()
    frames = np.asarray(frames, dtype=float)
    n_frames, radius = len(frames), float(np.linalg.norm(frames, axis=2).max())
    bond = float(np.linalg.norm(np.diff(frames[0], axis=0), axis=1).mean())
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background("white")
    with warnings.catch_warnings():                       # OSMesa builds swap FXAA for SSAA and warn
        warnings.simplefilter("ignore")
        pl.enable_anti_aliasing("fxaa")
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0, 0, 1)
    pl.camera.position = tuple(np.array([1.0, 0.55, 0.45]) / np.linalg.norm([1.0, 0.55, 0.45]) * radius * 2.7)
    pl.camera.view_angle = 30
    pl.open_movie(str(path), framerate=fps, quality=8)
    step = 360.0 * turns / max(n_frames, 1)
    with stage_bar(f"Video: {mode}", n_frames) as advance:
        for k, pts in enumerate(frames):
            s, cmap, clim = scalars(mode, pts, pl.camera.position, radius, comp)
            for name, mesh in _meshes(pv, pts, s, bond, representation).items():
                pl.add_mesh(mesh, name=name, scalars="s", cmap=cmap, clim=clim, specular=0.9, specular_power=45, ambient=0.3,
                            diffuse=0.8, smooth_shading=True, show_scalar_bar=False, reset_camera=False)
            pl.add_text(f"{LABEL[mode]}   |   frame {k + 1}/{n_frames}", position="lower_left", font_size=max(8, size // 55), color="#6b6485", name="label")
            pl.reset_camera_clipping_range()
            pl.write_frame()
            if poster is not None and k == n_frames // 2:
                pl.screenshot(str(poster))
            pl.camera.Azimuth(step)
            advance()
    pl.close()
    return Path(path)


def render_all(frames, outdir, modes, comp, fps, size, max_frames, representation="tube") -> dict:
    """Videos and posters for each colouring; returns {mode: (mp4, png)}; failures are logged, not raised."""
    outdir = Path(outdir)
    if len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    out = {}
    for mode in modes:
        try:
            mp4, png = outdir / f"structure_{mode}.mp4", outdir / f"structure_{mode}.png"
            render_video(frames, mp4, mode, comp, fps, size, poster=png, representation=representation)
            out[mode] = (mp4, png)
        except Exception as e:                                     # noqa: BLE001 - rendering is optional
            log.warning("Could not render the '%s' video: %s", mode, e)
    return out
