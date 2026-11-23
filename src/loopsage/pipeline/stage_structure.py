"""Stage: 3D structure diagnostics, smooth interpolated trajectory, interactive viewer and videos."""
import base64
import html
import os
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
    viewer: Path = None      # 3d_viewer.html (replica 0 when there are several)
    media: dict = None       # {colouring: (mp4, poster png)}
    replicas: list = None    # one StructureResult per replica (several replicas only; the metrics above are then pooled)


def enabled(cfg) -> bool:
    return bool(cfg.run_md)


def _modes(cfg, comp):
    """Requested colourings; compartments are added when a compartment track exists."""
    modes = [m for m in cfg.colorings if m != "compartments" or comp is not None]
    if comp is not None and "compartments" not in modes:
        modes.append("compartments")
    return modes


def _build(ctx, md, dirs) -> StructureResult:
    """Metrics, interactive viewer and videos of one MD run (written to that replica's folders)."""
    cfg, comp = ctx.cfg, ctx.inputs.comp
    log.info("Computing 3D structure metrics for %d frames", len(md.coords))
    res = StructureResult(compute_metrics(md.coords, cfg.contact_radius, comp))
    if not cfg.render_3d:
        return res
    frames = interpolate_trajectory(md.coords, cfg.interp_steps)          # aligned + smooth in time
    log.info("Interpolated %d MD frames to %d smooth frames; building the interactive 3D viewer", len(md.coords), len(frames))
    res.viewer = build_viewer(frames, dirs.viewer / "3d_viewer.html", f"LoopSage 3D - {ctx.geom.label}", comp, cfg.video_fps)
    if cfg.make_videos:
        try:
            from loopsage.viz3d.render import render_all
            log.info("Rendering 3D videos")
            res.media = render_all(frames, dirs.viewer, _modes(cfg, comp), comp, cfg.video_fps, cfg.video_size, cfg.max_video_frames, cfg.representation)
        except ImportError:
            log.warning("pyvista is not installed: skipped the videos (pip install pyvista imageio-ffmpeg)")
    return res


def compute(ctx):
    """One analysis per replica (own viewer and videos); with several, the metrics of all frames are pooled for the report."""
    parts = [_build(ctx, md, ctx.dirs.replica(k)) for k, md in enumerate(ctx.md.chains)]
    if len(parts) == 1:
        return parts[0]
    pooled = compute_metrics(ctx.md_ens.coords, ctx.cfg.contact_radius, ctx.inputs.comp)
    return StructureResult(pooled, parts[0].viewer, parts[0].media, parts)


def _snapshots(cfg, md, mc):
    """Start / middle / end renders, a 20-structure filmstrip and the MC and temperature bars (skipped without pyvista)."""
    from loopsage.molecular_dynamics.structures import align_frames
    picks = np.unique(np.linspace(0, len(md.coords) - 1, 20).round().astype(int))      # frames shown in the filmstrip
    if len(picks) < 3:
        return
    try:
        from loopsage.viz3d.render import render_stills
        imgs = render_stills(align_frames(np.asarray(md.coords, dtype=float))[picks], "rainbow", None, cfg.video_size, cfg.representation)
    except Exception as e:                                         # noqa: BLE001 - rendering is optional
        log.warning("Could not render the start/middle/end structures: %s", e)
        return
    structure_figs.snapshots_figure(imgs, picks, md, mc)


def _viewer_block(res, ctx):
    """Report card: embedded interactive viewer plus the videos (linked relative to the report)."""
    page = html.escape(res.viewer.read_text(encoding="utf-8"), quote=True)
    body = f'<iframe class="viewer" srcdoc="{page}" loading="lazy"></iframe>'
    rel = lambda f: Path(os.path.relpath(Path(f).resolve(), ctx.dirs.report.parent.resolve())).as_posix()
    if res.media:
        cells = []
        for mode, (mp4, png) in res.media.items():
            poster = base64.b64encode(Path(png).read_bytes()).decode()
            cells.append(f'<figure><video controls loop muted preload="metadata" poster="data:image/png;base64,{poster}" '
                         f'src="{rel(mp4)}"></video><figcaption>{LABEL[mode]}</figcaption></figure>')
        body += f'<div class="videos">{"".join(cells)}</div>'
    where = res.viewer.parent.resolve().relative_to(ctx.dirs.root.resolve()).as_posix()
    caption = ("Drag to rotate, wheel to zoom; the trajectory is aligned and smoothly interpolated between MD frames. "
               f"Standalone files: {where}/3d_viewer.html and the .mp4 videos"
               + (" (replica 0 shown; every replica has its own folder)." if ctx.replicas > 1 else "."))
    REGISTRY.add_block("Interactive 3D viewer and videos", body, caption, "structure_3d")


def _force_checks(ctx, m, md, mc, series=True):
    """Time series of the metrics plus bond/angle/excluded-volume/loop diagnostics."""
    from loopsage.analysis.structure import force_diagnostics, force_rows
    from loopsage.viz import structure_figs
    f = force_diagnostics(md, mc, ctx.cfg)
    emit_table("Force-field checks", ["Quantity", "Measured", "Expected", "Unit"],
               [[r[0], f"{r[1]:.4g}", f"{r[2]:.4g}", r[3]] for r in force_rows(f)],
               "Measured over all MD frames; 'expected' is the rest value or the thermal estimate.", "structure_3d")
    if series:                                                      # frames of several replicas are not one time series
        structure_figs.time_figure(m, md)
    structure_figs.bonds_angles_figure(f, md)
    structure_figs.excluded_volume_figure(f, md)
    structure_figs.loop_distance_figure(f, md)


def _figures(ctx, m, md, mc, series=True):
    """Shape, pair statistics, scaling, compartments and force-field checks of one set of frames."""
    structure_figs.shape_figure(m, md)
    structure_figs.pairs_figure(m, md)
    structure_figs.scaling_figure(m, md)
    if m.compart:
        structure_figs.compartment_figure_3d(m, md)
    _force_checks(ctx, m, md, mc, series)


def _replicas(ctx):
    """Table comparing the replicas; their own figures, snapshots, viewers and videos go to replica_<k> folders (not in the report)."""
    res = ctx.structure
    rows = [summary_rows(p.metrics) for p in res.replicas]
    emit_table("3D structure by replica", ["Metric", *[f"Replica {k}" for k in range(len(rows))], "Unit"],
               [[r[0], *[f"{x[i][1]:.4g}" for x in rows], r[3]] for i, r in enumerate(rows[0])],
               "Mean over the MD frames of each replica; replicas that agree reached the same kind of 3D structure.", "structure_3d")
    for k, (p, md, mc) in enumerate(zip(res.replicas, ctx.md.chains, ctx.mc.chains)):
        with REGISTRY.scope(ctx.dirs.replica(k).figures):
            _figures(ctx, p.metrics, md, mc)
            if ctx.cfg.render_3d:
                _snapshots(ctx.cfg, md, mc)


def describe(ctx):
    res, m, md = ctx.structure, ctx.structure.metrics, ctx.md_ens
    multi = ctx.replicas > 1
    rows = summary_rows(m)
    emit_table("3D structure diagnostics", ["Metric", "Mean", "Std", "Unit", "Meaning"],
               [[r[0], f"{r[1]:.4g}", f"{r[2]:.2g}", r[3], r[4]] for r in rows],
               "Averaged over the MD frames" + (" of all replicas" if multi else "") + " (pair-based metrics use an even subset of up to 60 frames). Lengths in nm.", "structure_3d")
    nu = float(rows[14][1])
    REGISTRY.fact("Radius of gyration", f"{rows[0][1]:.3g} nm")
    REGISTRY.fact("Fractal dimension", f"{1 / nu:.2f}" if nu == nu else "n/a")
    if res.viewer is not None:
        _viewer_block(res, ctx)
    if ctx.cfg.render_3d and not multi:
        _snapshots(ctx.cfg, md, ctx.mc_ens)
    _figures(ctx, m, md, ctx.mc_ens, series=not multi)
    if multi:
        _replicas(ctx)


if __name__ == "__main__":
    from loopsage.pipeline.runner import run_standalone
    run_standalone(NAME)
