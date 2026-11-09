"""Stage: 3D structure diagnostics, smooth interpolated trajectory, interactive viewer and videos."""
import base64
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from loopsage.analysis.structure import StructureMetrics, compute_metrics, summary_rows
from loopsage.common.utils import emit_table
from loopsage.logger import get_logger
from loopsage.molecular_dynamics.structures import interpolate_trajectory
from loopsage.reporting.report import REGISTRY
from loopsage.viz import structure_figs
from loopsage.viz3d.render import LABEL
from loopsage.viz3d.render import build_viewer

NAME, ATTR, CACHE = "structure", "structure", None
log = get_logger(__name__)


@dataclass
class StructureResult:
    metrics: StructureMetrics
    viewer: Path = None      # 3d_viewer.html
    media: dict = None       # {colouring: (mp4, poster png)}


def enabled(cfg) -> bool:
    return bool(cfg.run_md)


def _modes(cfg, comp):
    """Requested colourings; compartments are added when a compartment track exists."""
    modes = [m for m in cfg.colorings if m != "compartments" or comp is not None]
    if comp is not None and "compartments" not in modes:
        modes.append("compartments")
    return modes


def compute(ctx):
    cfg, md, comp = ctx.cfg, ctx.md, ctx.inputs.comp
    log.info("Computing 3D structure metrics for %d frames", len(md.coords))
    res = StructureResult(compute_metrics(md.coords, cfg.contact_radius, comp))
    if not cfg.render_3d:
        return res
    frames = interpolate_trajectory(md.coords, cfg.interp_steps)          # aligned + smooth in time
    log.info("Interpolated %d MD frames to %d smooth frames; building the interactive 3D viewer", len(md.coords), len(frames))
    res.viewer = build_viewer(frames, ctx.dirs.viewer / "3d_viewer.html", f"LoopSage 3D - {ctx.geom.label}", comp, cfg.video_fps)
    if cfg.make_videos:
        try:
            from loopsage.viz3d.render import render_all
            log.info("Rendering 3D videos")
            res.media = render_all(frames, ctx.dirs.viewer, _modes(cfg, comp), comp, cfg.video_fps, cfg.video_size, cfg.max_video_frames, cfg.representation)
        except ImportError:
            log.warning("pyvista is not installed: skipped the videos (pip install pyvista imageio-ffmpeg)")
    return res


def _snapshots(ctx, comp):
    """Start / middle / end renders, a 20-structure filmstrip and the MC and temperature bars (skipped without pyvista)."""
    from loopsage.molecular_dynamics.structures import align_frames
    md = ctx.md
    picks = np.unique(np.linspace(0, len(md.coords) - 1, 20).round().astype(int))      # frames shown in the filmstrip
    if len(picks) < 3:
        return
    try:
        from loopsage.viz3d.render import render_stills
        imgs = render_stills(align_frames(np.asarray(md.coords, dtype=float))[picks], "rainbow", None, ctx.cfg.video_size, ctx.cfg.representation)
    except Exception as e:                                         # noqa: BLE001 - rendering is optional
        log.warning("Could not render the start/middle/end structures: %s", e)
        return
    structure_figs.snapshots_figure(imgs, picks, md, ctx.mc)


def _viewer_block(res, ctx):
    """Report card: embedded interactive viewer plus the videos (linked relative to the report)."""
    page = html.escape(res.viewer.read_text(encoding="utf-8"), quote=True)
    body = f'<iframe class="viewer" srcdoc="{page}" loading="lazy"></iframe>'
    if res.media:
        cells = []
        for mode, (mp4, png) in res.media.items():
            poster = base64.b64encode(Path(png).read_bytes()).decode()
            cells.append(f'<figure><video controls loop muted preload="metadata" poster="data:image/png;base64,{poster}" '
                         f'src="../visualizations/3d/{Path(mp4).name}"></video><figcaption>{LABEL[mode]}</figcaption></figure>')
        body += f'<div class="videos">{"".join(cells)}</div>'
    caption = ("Drag to rotate, wheel to zoom; the trajectory is aligned and smoothly interpolated between MD frames. "
               "Standalone files: visualizations/3d/3d_viewer.html and the .mp4 videos.")
    REGISTRY.add_block("Interactive 3D viewer and videos", body, caption, "structure_3d")


def _force_checks(ctx, m, md):
    """Time series of the metrics plus bond/angle/excluded-volume/loop diagnostics."""
    from loopsage.analysis.structure import force_diagnostics, force_rows
    from loopsage.viz import structure_figs
    f = force_diagnostics(md, ctx.mc_ens, ctx.cfg)
    emit_table("Force-field checks", ["Quantity", "Measured", "Expected", "Unit"],
               [[r[0], f"{r[1]:.4g}", f"{r[2]:.4g}", r[3]] for r in force_rows(f)],
               "Measured over all MD frames; 'expected' is the rest value or the thermal estimate.", "structure_3d")
    structure_figs.time_figure(m, md)
    structure_figs.bonds_angles_figure(f, md)
    structure_figs.excluded_volume_figure(f, md)
    structure_figs.loop_distance_figure(f, md)


def describe(ctx):
    res, m, md = ctx.structure, ctx.structure.metrics, ctx.md
    rows = summary_rows(m)
    emit_table("3D structure diagnostics", ["Metric", "Mean", "Std", "Unit", "Meaning"],
               [[r[0], f"{r[1]:.4g}", f"{r[2]:.2g}", r[3], r[4]] for r in rows],
               "Averaged over the MD frames (pair-based metrics use an even subset of up to 60 frames). Lengths in nm.", "structure_3d")
    nu = float(rows[14][1])
    REGISTRY.fact("Radius of gyration", f"{rows[0][1]:.3g} nm")
    REGISTRY.fact("Fractal dimension", f"{1 / nu:.2f}" if nu == nu else "n/a")
    if res.viewer is not None:
        _viewer_block(res, ctx)
    if ctx.cfg.render_3d:
        _snapshots(ctx, ctx.inputs.comp)
    structure_figs.shape_figure(m, md)
    structure_figs.pairs_figure(m, md)
    structure_figs.scaling_figure(m, md)
    if m.compart:
        structure_figs.compartment_figure_3d(m, md)
    _force_checks(ctx, m, md)


if __name__ == "__main__":
    from loopsage.pipeline.runner import run_standalone
    run_standalone(NAME)
