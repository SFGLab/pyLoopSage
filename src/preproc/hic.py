"""Read a region of a .hic file (hic-straw) onto the bead grid of the model (same approach as LoCR).

The map is used as an optional reference: validation of the simulated maps and the target of --optimize.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from logger import get_logger

log = get_logger(__name__)
NORM_FALLBACK = ["KR", "SCALE", "VC_SQRT", "VC", "NONE"]
MAX_BINS = 6000                                        # largest matrix side read at once

# hic-straw can crash the interpreter on missing resolutions or normalisations, so every read runs
# in a separate process and a crash is just a failed trial.
_WORKER = """
import json, sys, numpy as np, hicstraw
path, mode = sys.argv[1], sys.argv[2]
hic = hicstraw.HiCFile(path)
if mode == "info":
    print(json.dumps({"res": [int(r) for r in hic.getResolutions()], "chroms": [c.name for c in hic.getChromosomes()]}))
else:
    chrom, res, nm, x0, x1, out = sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
    z = hic.getMatrixZoomData(chrom, chrom, "observed", nm, "BP", res)
    np.save(out, np.asarray(z.getRecordsAsMatrix(x0, x1, x0, x1), float))
"""


def have_hicstraw() -> bool:
    """True when hic-straw can be imported."""
    try:
        import hicstraw  # noqa: F401
        return True
    except ImportError:
        return False


def candidate_resolutions(available, bead_size: int, span: int, wanted=None) -> list:
    """Bin sizes to try, best first: the wanted one, then finer-or-equal sizes closest to the bead size, then coarser ones.
    Sizes needing more than MAX_BINS bins for the region are skipped."""
    av = sorted({int(r) for r in available if r > 0 and span / r <= MAX_BINS})
    pref = sorted(av, key=lambda r: (r > bead_size, bead_size % r != 0 if r <= bead_size else r % bead_size != 0, abs(np.log(r / bead_size))))
    return ([int(wanted)] if wanted and int(wanted) in av else []) + [r for r in pref if r != wanted]


def _chrom_name(names, chrom: str) -> str:
    """Chromosome name as the file spells it (chr1 or 1)."""
    for cand in (chrom, "chr" + chrom, chrom[3:] if chrom.startswith("chr") else chrom):
        if cand in names:
            return cand
    raise ValueError(f"chromosome {chrom} not in the .hic file (has: {', '.join(names[:8])} ...)")


def _run(*args, timeout=900) -> str:
    """Run the hic-straw worker; return stdout, or raise RuntimeError if it crashed, failed or timed out."""
    try:
        r = subprocess.run([sys.executable, "-c", _WORKER, *map(str, args)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("timeout")
    if r.returncode != 0:
        raise RuntimeError("crashed (segfault)" if r.returncode < 0 else (r.stderr.strip().splitlines() or ["error"])[-1])
    return r.stdout


def rebin(mat: np.ndarray, bin_bp: int, origin: int, geom):
    """Map an (m x m) matrix of `bin_bp` bins starting at `origin` bp onto the bead grid; also the valid fraction per bead.
    Finer data are summed into beads; coarser data are spread over the beads they cover (counts split by (bin/bead)^2)."""
    n, bs = geom.n_beads, geom.bp_per_bead
    valid = np.isfinite(mat).any(1)
    if bin_bp > bs:
        ctr = geom.start + (np.arange(n) + 0.5) * bs                          # bead centres
        idx = np.floor((ctr - origin) / bin_bp).astype(int)
        ok = (idx >= 0) & (idx < mat.shape[0])
        j = np.where(ok, idx, 0)
        out = mat[np.ix_(j, j)] / (bin_bp / bs) ** 2
        out[~ok, :] = np.nan
        out[:, ~ok] = np.nan
        return out, np.where(ok, valid[j], False).astype(float)
    ctr = origin + (np.arange(mat.shape[0]) + 0.5) * bin_bp
    idx = np.floor((ctr - geom.start) / bs).astype(int)
    ok = (idx >= 0) & (idx < n)
    m = np.nan_to_num(mat, nan=0.0)[np.ix_(ok, ok)]
    P = np.zeros((n, m.shape[0]))
    P[idx[ok], np.arange(m.shape[0])] = 1.0
    cnt, tot = P @ valid[ok].astype(float), P.sum(1)
    return P @ m @ P.T, np.divide(cnt, tot, out=np.zeros(n), where=tot > 0)


def load_hic_region(path: str, geom, norm: str = "KR", resolution=None, min_coverage: float = 0.5, max_trials: int = 12):
    """Hi-C map of the region on the bead grid (NaN = masked beads), by trial and error over resolutions and
    normalisations until one reads. Returns (map, info dict)."""
    if not have_hicstraw():
        raise ImportError("reading a .hic file needs hic-straw: pip install hic-straw")
    try:
        out = _run(path, "info")
        meta = json.loads(next(l for l in reversed(out.splitlines()) if l.startswith("{")))     # hic-straw may print other lines
    except (RuntimeError, StopIteration, ValueError) as e:
        raise ValueError(f"cannot open {path} with hic-straw: {e}")
    if not meta["chroms"] or not meta["res"]:
        raise ValueError(f"{path} does not look like a valid .hic file (no chromosomes or resolutions found)")
    chrom, span, bs = _chrom_name(meta["chroms"], geom.chrom), geom.end - geom.start, geom.bp_per_bead
    cands = candidate_resolutions(meta["res"], bs, span, resolution)
    if not cands:
        raise ValueError(f"no Hi-C resolution in the file suits a {span:,} bp region (file has {sorted(meta['res'])})")
    n_up = norm.upper()
    order = NORM_FALLBACK[NORM_FALLBACK.index(n_up):] if n_up in NORM_FALLBACK else [n_up, "NONE"]
    mat = used = res = o = None
    tmp, trials, fails = tempfile.mkdtemp(prefix="loopsage_hic_"), 0, []
    log.debug("hic: file resolutions %s; trying %s x %s", sorted(meta["res"]), cands[:6], order)
    for r in cands:
        x0, x1 = (geom.start // r) * r, -(-geom.end // r) * r - 1
        for nm in order:
            if trials >= max_trials:
                break
            trials += 1
            f = os.path.join(tmp, "m.npy")
            try:
                _run(path, "read", chrom, r, nm, x0, x1, f)
                a = np.load(f)
            except (RuntimeError, OSError, ValueError) as e:
                fails.append((r, nm, str(e)))
                log.debug("hic: trial %d: %d bp, %s failed: %s", trials, r, nm, e)
                continue
            if a.size and np.isfinite(a).any() and np.nansum(a) > 0:
                mat, used, res, o = a, nm, r, x0
                break
            fails.append((r, nm, "empty matrix"))
        if mat is not None or trials >= max_trials:
            break
    shutil.rmtree(tmp, ignore_errors=True)
    if mat is None:
        why = "; ".join(f"{r:,} bp {nm}: {w[:50]}" for r, nm, w in fails[:4])
        raise ValueError(f"no usable Hi-C matrix for {geom.label} after {trials} trials ({why})")
    if used != n_up:
        log.info("Hi-C normalisation %s not available, using %s", norm, used)
    if res > bs:
        log.info("Hi-C resolution %s bp is coarser than the bead size %s bp: finer detail is lost", f"{res:,}", f"{bs:,}")
    out, cover = rebin(mat, res, o, geom)
    out[cover < min_coverage] = np.nan
    out[:, cover < min_coverage] = np.nan
    return out, dict(resolution=int(res), norm=used, chrom=chrom, coverage=float((cover >= min_coverage).mean()), file=Path(path).name)


def hic_for_run(cfg, geom, folder):
    """(map, info) of the run's Hi-C file, cached in <folder>/hic_map.npy + hic_info.json; None when no --hic file is set."""
    if not getattr(cfg, "hic_file", None):
        return None
    folder = Path(folder)
    npy, js = folder / "hic_map.npy", folder / "hic_info.json"
    if npy.is_file() and js.is_file():
        info = json.loads(js.read_text())
        if info.get("file") == Path(cfg.hic_file).name and np.load(npy).shape == (geom.n_beads,) * 2:
            return np.load(npy), info
    mat, info = load_hic_region(cfg.hic_file, geom, cfg.hic_norm, cfg.hic_resolution or None)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(npy, mat)
    js.write_text(json.dumps(info, indent=1))
    return mat, info
