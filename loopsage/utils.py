#########################################################################
########### CREATOR: SEBASTIAN KORSAK, WARSAW 2022 ######################
#########################################################################

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr, kendalltau, mannwhitneyu, norm
from tqdm import tqdm
from .logger import get_logger


log = get_logger(__name__)

human_chromosome_lengths = {
    "hg19": {
        "chr1": 249_250_621,
        "chr2": 243_199_373,
        "chr3": 198_022_430,
        "chr4": 191_154_276,
        "chr5": 180_915_260,
        "chr6": 171_115_067,
        "chr7": 159_138_663,
        "chr8": 146_364_022,
        "chr9": 141_213_431,
        "chr10": 135_534_747,
        "chr11": 135_006_516,
        "chr12": 133_851_895,
        "chr13": 115_169_878,
        "chr14": 107_349_540,
        "chr15": 102_531_392,
        "chr16": 90_354_753,
        "chr17": 81_195_210,
        "chr18": 78_077_248,
        "chr19": 59_128_983,
        "chr20": 63_025_520,
        "chr21": 48_129_895,
        "chr22": 51_304_566,
        "chrX": 155_270_560,
        "chrY": 59_373_566,
        "chrM": 16_571,
    },

    "hg38": {
        "chr1": 248_956_422,
        "chr2": 242_193_529,
        "chr3": 198_295_559,
        "chr4": 190_214_555,
        "chr5": 181_538_259,
        "chr6": 170_805_979,
        "chr7": 159_345_973,
        "chr8": 145_138_636,
        "chr9": 138_394_717,
        "chr10": 133_797_422,
        "chr11": 135_086_622,
        "chr12": 133_275_309,
        "chr13": 114_364_328,
        "chr14": 107_043_718,
        "chr15": 101_991_189,
        "chr16": 90_338_345,
        "chr17": 83_257_441,
        "chr18": 80_373_285,
        "chr19": 58_617_616,
        "chr20": 64_444_167,
        "chr21": 46_709_983,
        "chr22": 50_818_468,
        "chrX": 156_040_895,
        "chrY": 57_227_415,
        "chrM": 16_569,
    },
}

def make_folder(folder_name):
    """
    Create the project directory structure if it does not already exist.
    """

    created = False

    # Main directory
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        created = True
        log.info(f"Created project directory: '{folder_name}'")

    elif os.path.isdir(folder_name):
        log.info(f"Using existing project directory: '{folder_name}'")

    else:
        log.error(
            f"Cannot create project directory. "
            f"A file named '{folder_name}' already exists."
        )
        raise IOError(
            f"File with name '{folder_name}' already exists. "
            "Please choose another project name."
        )

    # Required subdirectories
    for subfolder in ("plots", "other", "ensemble"):

        subfolder_path = os.path.join(folder_name, subfolder)

        if not os.path.isdir(subfolder_path):
            os.makedirs(subfolder_path, exist_ok=True)
            created = True
            log.info(f"Created subdirectory: '{subfolder_path}'")

    if created:
        log.info("Project directory structure is ready.")
    else:
        log.info("Project directory structure already exists.")

    return folder_name

############# Creation of mmcif and psf files #############
mmcif_atomhead = """data_nucsim
# 
_entry.id nucsim
# 
_audit_conform.dict_name       mmcif_pdbx.dic 
_audit_conform.dict_version    5.296 
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic 
# ----------- ATOMS ----------------
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_alt_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.pdbx_PDB_ins_code 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z
"""

mmcif_connecthead = """#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
"""

# ============================================================================
# Shared plot styling
# ============================================================================
 
SIM_COLOR = "#2563eb"
EXP_COLOR = "#dc2626"
NEUTRAL_COLOR = "#374151"
 
def _apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
        "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": "#e5e7eb", "grid.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
 
def _save_fig(fig, path, plot_name):
    fig.savefig(os.path.join(path, 'plots', f'{plot_name}.png'), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(path, 'plots', f'{plot_name}.pdf'), bbox_inches="tight")
    plt.close(fig)
 
def _report(msg, f=None):
    """Log a message and optionally mirror it into correlations.txt."""
    log.info(msg)
    if f is not None:
        f.write(msg + "\n")
 
def _pearson_ci(r, n, alpha=0.05):
    """95% CI for a Pearson r via Fisher z-transform."""
    if n < 4 or abs(r) >= 1:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    zcrit = norm.ppf(1 - alpha / 2)
    return np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
 
def _interpret_r(r):
    """Plain-language strength label (Cohen-style thresholds)."""
    a = abs(r)
    if a < 0.1: return "negligible"
    if a < 0.3: return "weak"
    if a < 0.5: return "moderate"
    if a < 0.7: return "strong"
    return "very strong"
 
def _minmax(mat):
    """Independent min-max scaling to [0, 1] - makes two signals/matrices on
    different native scales directly comparable (e.g. for a shared color
    scale or a y=x reference line)."""
    mat = np.asarray(mat, dtype=np.float64)
    lo, hi = np.nanmin(mat), np.nanmax(mat)
    return (mat - lo) / (hi - lo) if hi > lo else np.zeros_like(mat)
 
# ============================================================================
# Format detection & loading
# ============================================================================
 
def detect_interaction_format(interaction_file):
    """Auto-detect .bedpe / .bed / .narrowPeak from the file extension."""
    ext = os.path.splitext(interaction_file)[-1].lower()
    if ext in ('.bedpe', '.bed'):
        return ext[1:]
    if ext == '.narrowpeak':
        return 'narrowpeak'
    raise ValueError(f"Unsupported interaction file format '{ext}'. Expected .bedpe, .bed, or .narrowPeak.")
 
def _load_interaction_df(interaction_file, region, chrom, paired):
    df = pd.read_csv(interaction_file, sep='\t', header=None, comment='#')
    mask = (df[0] == chrom) & (df[1] >= region[0]) & (df[2] >= region[0]) & \
           (df[1] < region[1]) & (df[2] < region[1])
    if paired:
        mask &= (df[4] >= region[0]) & (df[5] >= region[0]) & \
                (df[4] < region[1]) & (df[5] < region[1])
    return df[mask].reset_index(drop=True)
 
def _get_weight_column(df, col_idx, default=1.0):
    if col_idx < df.shape[1]:
        return pd.to_numeric(df[col_idx], errors='coerce').fillna(default).values
    return np.full(len(df), default)
 
def _compute_bead_positions(starts, ends, region, resolution, N_beads):
    mid = (starts.values + ends.values) // 2
    beads = ((mid - region[0]) // resolution).astype(int)
    return np.clip(beads, 0, N_beads - 1)
 
def build_bedpe_pairs(df, region, resolution, N_beads, weight_col=6):
    """(x, y, weight) triplets - one per loop, x/y are anchor bead indices."""
    x = _compute_bead_positions(df[1], df[2], region, resolution, N_beads)
    y = _compute_bead_positions(df[4], df[5], region, resolution, N_beads)
    return list(zip(x, y, _get_weight_column(df, weight_col)))
 
def build_single_region_sites(df, region, resolution, N_beads, weight_col=4):
    """(x, weight) pairs - one per CTCF site, for .bed / .narrowPeak input."""
    x = _compute_bead_positions(df[1], df[2], region, resolution, N_beads)
    return list(zip(x, _get_weight_column(df, weight_col)))
 
# ============================================================================
# Signal construction (loop/site strength)
# ============================================================================
 
def build_strength_signal(mat_sim, df, N_beads, region=None, resolution=None):
    """
    Bead-level strength vectors for .bed/.narrowPeak: each site's score is
    placed on its bead (exp_vec), and the simulated signal (th_vec) is that
    bead's total row contact in mat_sim - there's no second anchor to look
    up a specific mat_sim[x, y] entry. (.bedpe uses compute_oe_bedpe_values
    + aggregate_to_beads instead, since it has anchor pairs to
    distance-normalize - see corr_exp_heat.)
    """
    exp_vec, th_vec = np.zeros(N_beads), np.zeros(N_beads)
    for x, w in build_single_region_sites(df, region, resolution, N_beads):
        exp_vec[x] += w
        th_vec[x] += mat_sim[x, :].sum() - mat_sim[x, x]
    return exp_vec, th_vec
 
def compute_oe_bedpe_values(df, mat_sim, region, resolution, N_beads):
    """
    Per-loop observed/expected (O/E) values: each loop's weight and simulated
    matrix value divided by the expected value at that anchor distance
    (from their respective P(s) curves) - removes the trivial "closer
    anchors interact more" confound before aggregation/correlation. Returns
    per-loop arrays (xs, ys, exp_oe, th_oe), nothing aggregated yet.
    """
    pairs = build_bedpe_pairs(df, region, resolution, N_beads)
    if not pairs:
        return (np.array([], dtype=int),) * 2 + (np.array([]),) * 2
 
    xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
    ws = np.array([p[2] for p in pairs], dtype=np.float64)
    dists = np.abs(ys - xs)
 
    exp_ps = compute_ps_curve_from_bedpe(df, region, resolution, N_beads)
    sim_ps = compute_ps_curve(mat_sim)
    sim_vals = np.array([mat_sim[x, y] for x, y in zip(xs, ys)], dtype=np.float64)
 
    eps = 1e-9
    return xs, ys, ws / (exp_ps[dists] + eps), sim_vals / (sim_ps[dists] + eps)
 
def aggregate_to_beads(xs, ys, exp_vals, th_vals, N_beads):
    """Mean- (not sum-) aggregate per-loop values onto each anchor's bead, so
    a bead touched by many loops isn't inflated relative to one strong loop."""
    exp_vec = np.zeros(N_beads); exp_n = np.zeros(N_beads)
    th_vec = np.zeros(N_beads); th_n = np.zeros(N_beads)
    for x, y, e, t in zip(xs, ys, exp_vals, th_vals):
        exp_vec[x] += e; exp_n[x] += 1; exp_vec[y] += e; exp_n[y] += 1
        th_vec[x] += t; th_n[x] += 1; th_vec[y] += t; th_n[y] += 1
    exp_vec = np.divide(exp_vec, exp_n, out=np.zeros_like(exp_vec), where=exp_n > 0)
    th_vec = np.divide(th_vec, th_n, out=np.zeros_like(th_vec), where=th_n > 0)
    return exp_vec, th_vec
 
def normalize_signal(vec, log_transform=True):
    """log1p (tames heavy right-skew so a few huge values can't dominate
    Pearson r) then z-score (comparable scale for plotting)."""
    vec = np.asarray(vec, dtype=np.float64)
    if log_transform:
        vec = np.log1p(np.clip(vec, a_min=0, a_max=None))
    std = np.std(vec)
    return (vec - np.mean(vec)) / std if std > 0 else vec - np.mean(vec)
 
def permutation_pvalue(exp_vec, th_vec, n_perm=500, seed=0):
    """Empirical (assumption-free) p-value for Pearson r via label permutation."""
    rng = np.random.default_rng(seed)
    observed, _ = pearsonr(exp_vec, th_vec)
    exceed = sum(abs(pearsonr(exp_vec, rng.permutation(th_vec))[0]) >= abs(observed) for _ in range(n_perm))
    return observed, (exceed + 1) / (n_perm + 1)
 
# ============================================================================
# Matrix-derived signals
# ============================================================================
 
def compute_ps_curve(mat):
    """Average contact strength vs. genomic distance (diagonal decay)."""
    return np.array([np.mean(np.diagonal(mat, offset=k)) for k in range(mat.shape[0])])
 
def compute_ps_curve_from_bedpe(df, region, resolution, N_beads):
    """Approximate experimental P(s) from loop weights binned by anchor distance."""
    ps_sum, ps_count = np.zeros(N_beads), np.zeros(N_beads)
    for x, y, w in build_bedpe_pairs(df, region, resolution, N_beads):
        d = abs(y - x)
        ps_sum[d] += w; ps_count[d] += 1
    return np.divide(ps_sum, ps_count, out=np.zeros_like(ps_sum), where=ps_count > 0)
 
def compute_insulation_score(mat, window=5):
    """Diamond insulation score around each bead."""
    N = mat.shape[0]
    ins = np.zeros(N)
    for i in range(window, N - window):
        ins[i] = np.mean(mat[i - window:i, i + 1:i + window + 1])
    return ins
 
def _valid_prob_vector(p):
    p = np.asarray(p, dtype=float)
    return np.all(np.isfinite(p) & ((p == -1) | ((p >= 0) & (p <= 1))))
 
def _extract_reverse_bias_prob(df, col):
    """Probability the site's best hit is reverse-oriented ("<"); None if invalid/missing."""
    if col >= df.shape[1]:
        return None
    p = pd.to_numeric(df[col], errors='coerce').fillna(-1).values
    return p if _valid_prob_vector(p) else None
 
def _local_asymmetry(mat, bead, window):
    """Right-minus-left simulated contact strength around a bead."""
    N = mat.shape[0]
    lo, hi = max(0, bead - window), min(N, bead + window + 1)
    return mat[bead, bead + 1:hi].sum() - mat[bead, lo:bead].sum()
 
def _collect_orientation_probs(df, fmt, region, resolution, N_beads):
    """Per-site (bead, reverse-bias probability) pairs, across all formats."""
    beads, probs = [], []
    if fmt == 'bedpe':
        for col, coords in ((7, (1, 2)), (8, (4, 5))):
            p = _extract_reverse_bias_prob(df, col)
            if p is not None:
                beads.extend(_compute_bead_positions(df[coords[0]], df[coords[1]], region, resolution, N_beads))
                probs.extend(p)
    elif fmt == 'bed':
        p = _extract_reverse_bias_prob(df, 6)
        if p is not None:
            beads.extend(_compute_bead_positions(df[1], df[2], region, resolution, N_beads))
            probs.extend(p)
    elif fmt == 'narrowpeak':
        p_fwd = _extract_reverse_bias_prob(df, 5)
        if p_fwd is not None:
            beads.extend(_compute_bead_positions(df[1], df[2], region, resolution, N_beads))
            probs.extend(np.where(p_fwd >= 0, 1 - p_fwd, -1.0))
    return np.array(beads), np.array(probs)
 
# ============================================================================
# Correlation reporting (logging + file + plot)
# ============================================================================
 
def _plot_signal_comparison(exp_vec, th_vec, pears, spear, signal_name, path, plot_name, xlabel):
    """Experimental trace, simulated trace, and a min-max-normalized regression
    scatter (with a y=x reference line) of the two."""
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1.1], hspace=0.35, wspace=0.28)
    ax_exp = fig.add_subplot(gs[0, 0])
    ax_sim = fig.add_subplot(gs[1, 0], sharex=ax_exp)
    ax_scatter = fig.add_subplot(gs[:, 1])
 
    x_idx = np.arange(len(exp_vec))
    ax_exp.plot(x_idx, exp_vec, color=EXP_COLOR, lw=1.6)
    ax_exp.fill_between(x_idx, exp_vec, color=EXP_COLOR, alpha=0.12)
    ax_exp.set_ylabel("Experimental signal")
    ax_exp.set_title(f"{signal_name}: experimental vs. simulated", fontsize=15, loc="left")
    plt.setp(ax_exp.get_xticklabels(), visible=False)
 
    ax_sim.plot(x_idx, th_vec, color=SIM_COLOR, lw=1.6)
    ax_sim.fill_between(x_idx, th_vec, color=SIM_COLOR, alpha=0.12)
    ax_sim.set_ylabel("Simulated signal")
    ax_sim.set_xlabel(xlabel)
 
    # Min-max both signals to [0, 1] for this panel only: exp_vec/th_vec live
    # on different native scales, so a y=x "perfect agreement" line only
    # means something once both are on the same [0, 1] scale.
    exp_s, th_s = _minmax(exp_vec), _minmax(th_vec)
    ax_scatter.scatter(exp_s, th_s, s=20, color=NEUTRAL_COLOR, alpha=0.55,
                        edgecolor="white", linewidth=0.3, zorder=3)
    ax_scatter.plot([0, 1], [0, 1], color="#9ca3af", lw=1.3, linestyle=":",
                     label="y = x (perfect agreement)", zorder=1)
    if len(exp_s) > 1 and np.std(exp_s) > 0:
        coeffs = np.polyfit(exp_s, th_s, 1)
        xs = np.linspace(0, 1, 100)
        ax_scatter.plot(xs, np.polyval(coeffs, xs), color="#111111", lw=2, linestyle="--",
                         label="Best fit", zorder=2)
    ax_scatter.set_xlim(-0.02, 1.02); ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.set_xlabel("Experimental (min-max normalized)")
    ax_scatter.set_ylabel("Simulated (min-max normalized)")
    ax_scatter.set_title("Correlation", fontsize=13)
    ax_scatter.legend(loc="lower right", fontsize=9, frameon=False)
    ax_scatter.text(0.05, 0.95, f"Pearson r = {pears:.3f}\nSpearman \u03c1 = {spear:.3f}",
                     transform=ax_scatter.transAxes, va="top", ha="left", fontsize=11,
                     bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d1d5db"))
 
    _save_fig(fig, path, plot_name)
 
def report_correlation(exp_vec, th_vec, label, signal_name, f, path=None,
                        plot_name=None, xlabel="Genomic distance (simulation beads)",
                        n_perm=0, seed=0, drop_zero_pairs=True, summary=None,
                        mask_exp=None, mask_th=None):
    """
    Pearson/Spearman/Kendall correlation between two 1D signals.
 
    By default (drop_zero_pairs=True) positions where BOTH signals are
    exactly zero are excluded before correlating: a bead/loop with no signal
    on either side "agrees" trivially and inflates the correlation without
    reflecting anything real. Zero-detection uses `mask_exp`/`mask_th` if
    given (the RAW, pre-normalization vectors) rather than `exp_vec`/`th_vec`
    themselves - after log1p + z-scoring, a true raw zero is shifted away
    from exactly 0, so checking normalized values would almost never find
    any zeros to drop. Reports n (and how many pairs were dropped), R\u00b2,
    a 95% CI on Pearson r, a plain-language strength label, and optionally a
    permutation p-value. Appends a row to `summary` if given. Returns the
    Pearson coefficient, or None if too little/mismatched data.
    """
    exp_vec, th_vec = np.asarray(exp_vec, dtype=np.float64), np.asarray(th_vec, dtype=np.float64)
    if len(exp_vec) != len(th_vec):
        log.warning(f"Mismatched vector lengths for {label} correlation ({signal_name}) - skipping.")
        return None
 
    n_total = len(exp_vec)
    if drop_zero_pairs:
        me = np.asarray(mask_exp, dtype=np.float64) if mask_exp is not None else exp_vec
        mt = np.asarray(mask_th, dtype=np.float64) if mask_th is not None else th_vec
        keep = ~((me == 0) & (mt == 0))
        exp_vec, th_vec = exp_vec[keep], th_vec[keep]
    n_used = len(exp_vec)
 
    if n_used < 3:
        log.warning(f"Not enough data for {label} correlation ({signal_name}) after filtering - skipping.")
        return None
 
    pears, pval1 = pearsonr(th_vec, exp_vec)
    spear, pval2 = spearmanr(th_vec, exp_vec)
    kendal, pval3 = kendalltau(th_vec, exp_vec)
    ci_lo, ci_hi = _pearson_ci(pears, n_used)
 
    _report(f'-------------- {label} Correlation ({signal_name}) -----------')
    dropped = f"  [excluded {n_total - n_used}/{n_total} zero/zero pairs ({100*(n_total - n_used)/n_total:.1f}%)]" \
              if drop_zero_pairs and n_used < n_total else ""
    _report(f"n = {n_used}{dropped}", f)
    _report(f"Pearson r = {pears:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  R\u00b2 = {pears**2:.3f}  "
            f"(p={pval1:.3g}, {_interpret_r(pears)})", f)
    _report(f"Spearman \u03c1 = {spear:.3f} (p={pval2:.3g})   Kendall \u03c4 = {kendal:.3f} (p={pval3:.3g})", f)
 
    if n_perm > 0:
        _, perm_pval = permutation_pvalue(exp_vec, th_vec, n_perm=n_perm, seed=seed)
        _report(f"Permutation test ({n_perm} shuffles) empirical p-value: {perm_pval:.4f}", f)
    if f is not None:
        f.write('\n')
 
    if summary is not None:
        summary.append({"label": label, "signal": signal_name, "r": pears, "r2": pears**2, "n": n_used})
 
    if plot_name and path is not None:
        _plot_signal_comparison(exp_vec, th_vec, pears, spear, signal_name, path, plot_name, xlabel)
    return pears
 
# ============================================================================
# Heatmap-vs-heatmap comparison plots (need a full experimental matrix)
# ============================================================================
 
def plot_heatmaps_side_by_side(mat_sim, mat_exp, path, plot_name="heatmaps_side_by_side", cmap="Reds"):
    """Experimental and simulated maps side by side, each independently
    min-max normalized to [0, 1] so both share a fair, comparable color scale."""
    sim, exp = _minmax(np.log1p(mat_sim)), _minmax(np.log1p(mat_exp))
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    axs[0].imshow(exp, cmap=cmap, vmin=0, vmax=1, origin="lower"); axs[0].set_title("Experimental", fontweight="bold")
    im1 = axs[1].imshow(sim, cmap=cmap, vmin=0, vmax=1, origin="lower"); axs[1].set_title("Simulated", fontweight="bold")
    for ax in axs:
        ax.set_xlabel("Bead index"); ax.grid(False)
    axs[0].set_ylabel("Bead index")
    fig.colorbar(im1, ax=axs, shrink=0.8, label="Normalized contact strength (0-1)")
    fig.suptitle("Simulated vs. Experimental Contact Maps", fontsize=16, fontweight="bold")
    _save_fig(fig, path, plot_name)
 
def plot_matrix_triangle_comparison(mat_sim, mat_exp, path, plot_name="matrix_triangle_comparison",
                                     cmap="Reds", pears=None):
    """
    Hi-C-style split heatmap: experimental upper triangle, simulated lower.
    Both matrices are independently min-max normalized to [0, 1] first,
    since they're different quantities on different scales - a shared raw
    color scale (or treating the diagonal as an x=y identity line) wouldn't
    be meaningful otherwise. The diagonal is now just a thin visual
    separator between the two triangles, not a value-equality reference.
    """
    N = mat_sim.shape[0]
    sim, exp = _minmax(np.log1p(mat_sim)), _minmax(np.log1p(mat_exp))
 
    combined = np.full_like(sim, np.nan)
    iu, il = np.triu_indices(N, k=1), np.tril_indices(N, k=-1)
    combined[iu] = exp[iu]; combined[il] = sim[il]
 
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(combined, cmap=cmap, vmin=0, vmax=1, origin="lower")
    ax.plot([0, N - 1], [0, N - 1], color="black", lw=0.7, alpha=0.35)  # boundary only
    ax.grid(False)
    ax.text(0.98, 0.03, "Simulated", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")
    ax.text(0.02, 0.97, "Experimental", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
 
    title = "Simulated vs. Experimental (triangle split, each 0-1 normalized)"
    if pears is not None:
        title += f"  \u2022  Pearson r = {pears:.3f}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Bead index"); ax.set_ylabel("Bead index")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized contact strength (0-1)")
    _save_fig(fig, path, plot_name)
 
def plot_difference_heatmap(mat_sim, mat_exp, path, plot_name="difference_heatmap", cmap="RdBu_r"):
    """Symmetric log2-ratio map: red = simulated-enriched, blue = experimental-enriched."""
    eps = 1e-6
    ratio = np.log2((mat_sim + eps) / (mat_exp + eps))
    vmax = np.nanpercentile(np.abs(ratio), 99)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ratio, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
    ax.grid(False)
    ax.set_title("Simulated vs. Experimental (log\u2082 ratio)", fontweight="bold")
    ax.set_xlabel("Bead index"); ax.set_ylabel("Bead index")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log\u2082(simulated / experimental)")
    _save_fig(fig, path, plot_name)
 
# ============================================================================
# Suggested diagnostics for peak/anchor-only data (logger-reported)
# ============================================================================
 
def report_orientation_canonicity_consistency(mat_sim, df, region, resolution, N_beads, f):
    """Mann-Whitney test: do canonical (forward-left/reverse-right) .bedpe
    loops show higher simulated contact strength than non-canonical ones?"""
    p1, p2 = _extract_reverse_bias_prob(df, 7), _extract_reverse_bias_prob(df, 8)
    if p1 is None or p2 is None:
        _report("Skipping canonical-orientation check: needs valid prob1/prob2 columns.", f)
        return
 
    pairs = build_bedpe_pairs(df, region, resolution, N_beads)
    values = np.array([mat_sim[x, y] for x, y, w in pairs])
    valid = (p1 >= 0) & (p2 >= 0)
    canonical = valid & (p1 < 0.4) & (p2 > 0.6)
    noncanonical = valid & ~canonical
 
    if canonical.sum() < 3 or noncanonical.sum() < 3:
        _report(f"Skipping canonical-orientation check: too few loops "
                f"(canonical={int(canonical.sum())}, non-canonical={int(noncanonical.sum())}).", f)
        return
 
    v_can, v_non = values[canonical], values[noncanonical]
    _, pval = mannwhitneyu(v_can, v_non, alternative='greater')
 
    _report('-------------- Suggested diagnostic: canonical-orientation consistency -----------')
    _report(f"Canonical: n={int(canonical.sum())}, mean={np.mean(v_can):.4f}  |  "
            f"Non-canonical: n={int(noncanonical.sum())}, mean={np.mean(v_non):.4f}  |  "
            f"Mann-Whitney p={pval:.3e}\n", f)
 
def report_orientation_asymmetry_consistency(mat_sim, df, fmt, region, resolution, N_beads, window, f):
    """Correlates each site's orientation probability against the local L/R
    asymmetry of the simulated matrix. Expected sign: negative."""
    beads, probs = _collect_orientation_probs(df, fmt, region, resolution, N_beads)
    valid = probs >= 0
    if valid.sum() < 5:
        _report("Skipping orientation-vs-asymmetry check: too few sites with known orientation.", f)
        return
 
    asymmetry = np.array([_local_asymmetry(mat_sim, b, window) for b in beads[valid]])
    corr, pval = pearsonr(probs[valid], asymmetry)
 
    _report('-------------- Suggested diagnostic: orientation vs. simulated asymmetry -----------')
    _report(f"n={int(valid.sum())}  Pearson r={corr:.3f} (p={pval:.3e})  Expected sign: negative.\n", f)
 
# ============================================================================
# Main entry point
# ============================================================================
 
def corr_exp_heat(mat_sim, interaction_file, region, chrom, N_beads, path,
                   mat_exp=None, insulation_window=5):
    """
    Correlate the simulated contact matrix against experimental signal(s).
    Format (.bedpe/.bed/.narrowPeak) is auto-detected from the extension.
 
    Loop/site strength is always computed, with two corrections applied to
    .bedpe: distance (O/E) normalization before anything else (removes the
    "closer anchors interact more" confound), and mean- (not sum-)
    aggregation onto beads, plus a "Direct" per-loop variant with no bead
    aggregation at all (the most reliable of the three). Every signal pair
    is normalized (log1p + z-score) before correlating. Positions where BOTH
    signals are exactly zero are excluded by default in every correlation
    (they trivially "agree" and inflate r without reflecting anything real);
    the loop/site-strength metric also reports a zeros-included reference
    value alongside, so you can see how much that filtering actually
    mattered. A permutation p-value backs up the main correlations, and a
    summary table with R\u00b2 and 95% CIs is printed at the end.
 
    P(s) decay and the insulation-score proxy only run for .bedpe (or with
    `mat_exp` for P(s)), since .bed/.narrowPeak alone lack anchor pairs or a
    full matrix. Two logger-only diagnostics use orientation data already in
    the file. Heatmap-vs-heatmap plots require `mat_exp`.
 
    Returns
    -------
    pears : float
        Pearson correlation of the primary (zero-excluded) loop/site-strength signal.
    """
    fmt = detect_interaction_format(interaction_file)
    log.info(f"Detected interaction file format: {fmt}")
 
    _apply_plot_style()
    os.makedirs(os.path.join(path, 'other'), exist_ok=True)
    os.makedirs(os.path.join(path, 'plots'), exist_ok=True)
    corr_path = os.path.join(path, 'other', 'correlations.txt')
 
    resolution = max((region[1] - region[0]) // N_beads, 1)
    df = _load_interaction_df(interaction_file, region, chrom, paired=(fmt == 'bedpe'))
    summary = []
 
    with open(corr_path, "w") as f:
        _report("=" * 70, f)
        _report("CORRELATION REPORT", f)
        _report("=" * 70, f)
        _report(f"Format: {fmt}   Region: {chrom}:{region[0]}-{region[1]}   "
                 f"N_beads: {N_beads}   Resolution: {resolution} bp/bead", f)
        _report("Zero/zero pairs are excluded by default in each metric below "
                 "(noted per-metric); a zeros-included reference is also shown "
                 "for the main loop/site-strength metric.\n", f)
 
        # 1. Loop / site strength
        if fmt == 'bedpe':
            _report("Applying distance (O/E) normalization for .bedpe before aggregation/correlation.", f)
            xs, ys, exp_oe, th_oe = compute_oe_bedpe_values(df, mat_sim, region, resolution, N_beads)
            exp_vec, th_vec = aggregate_to_beads(xs, ys, exp_oe, th_oe, N_beads)
            strength_label = "loop strength (O/E, distance-normalized)"
        else:
            exp_vec, th_vec = build_strength_signal(mat_sim, df, N_beads, region, resolution)
            strength_label = "loop/site strength"
 
        exp_n, th_n = normalize_signal(exp_vec), normalize_signal(th_vec)
 
        # Primary: zero/zero beads excluded (recommended, plotted). Zero
        # detection uses the RAW exp_vec/th_vec (mask_exp/mask_th), since
        # exp_n/th_n are already log1p+z-scored and a true raw zero rarely
        # survives as exactly 0 after that transform.
        pears = report_correlation(exp_n, th_n, "Primary (zero-excluded)", strength_label,
                                    f, path, "pearson", n_perm=500, drop_zero_pairs=True,
                                    mask_exp=exp_vec, mask_th=th_vec, summary=summary)
        # Reference: same data with zero/zero beads left in, for contrast only (no plot)
        report_correlation(exp_n, th_n, "Reference (zeros included)", strength_label,
                            f, drop_zero_pairs=False, summary=summary)
 
        if fmt == 'bedpe' and len(xs) > 1:
            report_correlation(normalize_signal(exp_oe), normalize_signal(th_oe),
                                "Direct (recommended)", "per-loop strength (O/E), no bead aggregation",
                                f, path, "pearson_direct", xlabel="Loop index", n_perm=500,
                                mask_exp=exp_oe, mask_th=th_oe, summary=summary)
 
        # 2. P(s) diagonal decay
        sim_ps = compute_ps_curve(mat_sim)
        if fmt == 'bedpe':
            exp_ps = compute_ps_curve_from_bedpe(df, region, resolution, N_beads)
            report_correlation(normalize_signal(exp_ps), normalize_signal(sim_ps),
                                "P(s) decay", "diagonal decay", f, path, "ps_decay",
                                mask_exp=exp_ps, mask_th=sim_ps, summary=summary)
        elif mat_exp is not None:
            exp_ps_full = compute_ps_curve(mat_exp)
            report_correlation(normalize_signal(exp_ps_full), normalize_signal(sim_ps),
                                "P(s) decay", "diagonal decay", f, path, "ps_decay",
                                mask_exp=exp_ps_full, mask_th=sim_ps, summary=summary)
        else:
            _report("Skipping P(s) decay correlation: needs .bedpe loop distances or `mat_exp`.", f)
 
        # 3. Insulation score proxy (bedpe only)
        if fmt == 'bedpe':
            _report("Note: insulation-score correlation below is an approximate proxy "
                    "(loop-anchor O/E density vs. simulated insulation score), not a strict validation.", f)
            ins_sim = compute_insulation_score(mat_sim, insulation_window)
            report_correlation(exp_n, normalize_signal(ins_sim),
                                "Boundary proxy", "anchor density vs. simulated insulation score",
                                f, path, "insulation_proxy",
                                mask_exp=exp_vec, mask_th=ins_sim, summary=summary)
        else:
            _report("Skipping insulation-score proxy: only meaningful for .bedpe input.", f)
 
        # 4. Orientation-based diagnostics
        if fmt == 'bedpe':
            report_orientation_canonicity_consistency(mat_sim, df, region, resolution, N_beads, f)
        report_orientation_asymmetry_consistency(mat_sim, df, fmt, region, resolution, N_beads,
                                                  insulation_window, f)
 
        # 5. Heatmap-vs-heatmap comparisons
        if mat_exp is not None:
            plot_heatmaps_side_by_side(mat_sim, mat_exp, path)
            plot_matrix_triangle_comparison(mat_sim, mat_exp, path, pears=pears)
            plot_difference_heatmap(mat_sim, mat_exp, path)
        else:
            _report("Skipping heatmap-vs-heatmap comparison plots: pass `mat_exp` to enable these.", f)
 
        # 6. Summary table
        _report("=" * 70, f)
        _report("SUMMARY", f)
        _report("=" * 70, f)
        for row in summary:
            _report(f"{row['label']:<26} {row['signal']:<48} "
                     f"r={row['r']:+.3f}  R\u00b2={row['r2']:.3f}  n={row['n']:<6} "
                     f"({_interpret_r(row['r'])})", f)
        _report("=" * 70, f)
 
    return pears

def write_cmm(comps,name):
    comp_old = 2
    counter, start = 0, 0
    comp_dict = {-1:'red', 1:'blue'}
    content = ''

    for i, comp in enumerate(comps):
        if comp_old==comp:
            counter+=1
        elif i!=0:
            content+=f'color {comp_dict[comp_old]} :{start}-{start+counter+1}\n'
            counter, start = 0, i
        comp_old=comp

    content+=f'color {comp_dict[comp]} :{start}-{start+counter+1}\n'
    with open(name, 'w') as f:
        f.write(content)

def write_mmcif(points,cif_file_name='LE_init_struct.cif'):
    atoms = ''
    n = len(points)
    for i in range(0,n):
        x = points[i][0]
        y = points[i][1]
        try:
            z = points[i][2]
        except IndexError:
            z = 0.0
        atoms += ('{0:} {1:} {2:} {3:} {4:} {5:} {6:}  {7:} {8:} '
                '{9:} {10:.3f} {11:.3f} {12:.3f}\n'.format('ATOM', i+1, 'D', 'CA',\
                                                            '.', 'ALA', 'A', 1, i+1, '?',\
                                                            x, y, z))

    connects = ''
    for i in range(0,n-1):
        connects += f'C{i+1} covale ALA A {i+1} CA ALA A {i+2} CA\n'

    # Save files
    ## .pdb
    cif_file_content = mmcif_atomhead+atoms+mmcif_connecthead+connects

    with open(cif_file_name, 'w') as f:
        f.write(cif_file_content)

def generate_psf(n: int, file_name='stochastic_LE.psf', title="No title provided"):
    """
    Saves PSF file. Useful for trajectories in DCD file format.
    :param n: number of points
    :param file_name: PSF file name
    :param title: Human readable string. Required in PSF file.
    :return: List with string records of PSF file.
    """
    assert len(title) < 40, "provided title in psf file is too long."
    # noinspection PyListCreation
    lines = ['PSF CMAP\n']
    lines.append('\n')
    lines.append('      1 !NTITLE\n')
    lines.append('REMARKS {}\n'.format(title))
    lines.append('\n')
    lines.append('{:>8} !NATOM\n'.format(n))
    for k in range(1, n + 1):
        lines.append('{:>8} BEAD {:<5} ALA  CA   A      0.000000        1.00 0           0\n'.format(k, k))
    lines.append('\n')
    lines.append('{:>8} !NBOND: bonds\n'.format(n - 1))
    for i in range(1, n):
        lines.append('{:>8}{:>8}\n'.format(i, i + 1))
    with open(file_name, 'w') as f:
        f.writelines(lines)

############# Computation of heatmaps #############
def get_coordinates_pdb(file):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly. 
    
    Input:
    file (str): the path of the .pdb file.
    
    Output:
    V (np.array): the matrix of coordinates
    '''
    V = list()
    
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("CONNECT") or line.startswith("END") or line.startswith("TER"):
                break
            if line.startswith("HETATM"): 
                x = float(line[31:38])
                y = float(line[39:46])
                z = float(line[47:54])
                V.append([x, y, z])
    
    return np.array(V)

def get_coordinates_cif(file):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly. 
    
    Input:
    file (str): the path of the .cif file.
    
    Output:
    V (np.array): the matrix of coordinates
    '''
    V = list()
    
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("ATOM"):
                columns = line.split()
                x = eval(columns[10])
                y = eval(columns[11])
                z = eval(columns[12])
                V.append([x, y, z])
    
    return np.array(V)

def get_coordinates_mm(mm_vec):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly. 
    
    Input:
    file (Openmm Qunatity): an OpenMM vector of the form 
    Quantity(value=[Vec3(x=0.16963918507099152, y=0.9815883636474609, z=-1.4776774644851685), 
    Vec3(x=0.1548253297805786, y=0.9109517931938171, z=-1.4084612131118774), 
    Vec3(x=0.14006929099559784, y=0.8403329849243164, z=-1.3392155170440674), 
    Vec3(x=0.12535107135772705, y=0.7697405219078064, z=-1.269935131072998),
    ...,
    unit=nanometer)
    
    Output:
    V (np.array): the matrix of coordinates
    '''
    V = list()

    for i in range(len(mm_vec)):
        x, y ,z = mm_vec[i][0]._value, mm_vec[i][1]._value, mm_vec[i][2]._value
        V.append([x, y, z])
    
    return np.array(V)

def get_heatmap(mm_vec, save_path=None, contact_radius=1.0, alpha=3.0,
                 width=0.2, eps=1e-3, mode='gaussian', transform=None, save=False):
    '''
    Convert one frame of 3D bead coordinates into a Hi-C-like contact map.

    contact_radius : float
        Characteristic contact distance, same units as mm_vec (nm) - the
        length scale at which two beads count as "in contact". Should match
        your force field's bead size / excluded-volume radius, not be left
        as an arbitrary constant.
    alpha : float
        Decay exponent, used by 'soft' and 'power_law' modes. Real Hi-C
        P(s) is roughly a power law with alpha ~3-4 for an ideal-chain-like
        polymer (lower alpha = more long-range contacts, e.g. ~1 for a
        fractal globule).
    width : float
        Transition width for 'sigmoid' mode (smaller = closer to 'binary').
    eps : float
        Small distance floor added before any 1/d-style kernel ('power_law',
        'inverse', 'inverse_squared'), so beads at zero distance (the
        diagonal) don't produce +inf. Choose it relative to contact_radius
        (e.g. contact_radius * 1e-3) rather than leaving the raw default,
        since it sets the ceiling value those kernels can reach.
    mode : {'binary', 'soft', 'power_law', 'exponential', 'gaussian', 'sigmoid', 'inverse', 'inverse_squared'}
        The contact-probability kernel applied to the raw pairwise distance
        matrix:
          'binary'          -> 1 if d <= contact_radius else 0
                                (classic thresholded contact map)
          'soft'            -> 1 / (1 + (d / contact_radius)**alpha)
                                (bounded in [0,1], flattens near d=0 - good general default)
          'power_law'       -> (d / contact_radius + eps)**(-alpha)
                                (unbounded near d=0, matches the classic Hi-C P(s) ~ s^-alpha
                                scaling directly - useful when comparing/fitting decay exponents)
          'exponential'     -> exp(-d / contact_radius)
                                (simple exponential contact decay)
          'gaussian'        -> exp(-d**2 / (2 * contact_radius**2))
                                (smooth Gaussian-chain-like kernel; falls off faster at long
                                range than 'exponential')
          'sigmoid'         -> 1 / (1 + exp((d - contact_radius) / width))
                                (smoothed version of 'binary' - tunable transition sharpness
                                via `width` instead of a hard cutoff)
          'inverse'         -> 1 / (d + eps)
                                (the original naive kernel - kept for backward compatibility/
                                comparison; decays far more slowly than real Hi-C P(s), so
                                prefer 'soft' or 'power_law' for anything quantitative)
          'inverse_squared' -> 1 / (d**2 + eps)
                                (steeper naive falloff than 'inverse', still not P(s)-calibrated)
    transform : {None, 'log1p', 'sqrt', 'zscore', 'minmax'}
        Optional post-hoc transform applied to the contact matrix AFTER the
        kernel above - independent of `mode`, since "how do two beads count
        as a contact" and "how do I want to view/compare the result" are
        separate questions:
          'log1p'  -> log(1 + mat)   (dynamic-range compression, common before plotting)
          'sqrt'   -> sqrt(mat)      (milder compression than log1p)
          'zscore' -> (mat - mean) / std   (useful before correlating against a
                                             z-scored experimental map)
          'minmax' -> rescale to [0, 1]
    '''
    V = get_coordinates_mm(mm_vec)

    if not np.all(np.isfinite(V)):
        raise ValueError(
            "Non-finite coordinates detected (NaN/Inf) - the simulation "
            "likely became numerically unstable. Refusing to build a "
            "heatmap from this frame rather than silently corrupting the average."
        )

    # pdist + squareform instead of cdist(V, V, ...): only computes the
    # N*(N-1)/2 unique pairwise distances instead of the full redundant matrix.
    d = distance.squareform(distance.pdist(V, 'euclidean'))

    if mode == 'binary':
        mat = (d <= contact_radius).astype(np.float64)
    elif mode == 'soft':
        mat = 1.0 / (1.0 + (d / contact_radius) ** alpha)
    elif mode == 'power_law':
        mat = (d / contact_radius + eps) ** (-alpha)
    elif mode == 'exponential':
        mat = np.exp(-d / contact_radius)
    elif mode == 'gaussian':
        mat = np.exp(-(d ** 2) / (2 * contact_radius ** 2))
    elif mode == 'sigmoid':
        mat = 1.0 / (1.0 + np.exp((d - contact_radius) / width))
    elif mode == 'inverse':
        mat = 1.0 / (d + eps)
    elif mode == 'inverse_squared':
        mat = 1.0 / (d ** 2 + eps)
    else:
        raise ValueError(
            f"Unknown mode '{mode}': expected one of 'binary', 'soft', "
            f"'power_law', 'exponential', 'gaussian', 'sigmoid', 'inverse', "
            f"'inverse_squared'."
        )
    # Note: every kernel above naturally evaluates at its maximum on the
    # diagonal (d=0), so self-contact is already the strongest entry in the
    # matrix without any special-casing - no extra diagonal fill-in needed.

    if transform == 'log1p':
        mat = np.log1p(mat)
    elif transform == 'sqrt':
        mat = np.sqrt(np.clip(mat, a_min=0, a_max=None))
    elif transform == 'zscore':
        std = np.std(mat)
        mat = (mat - np.mean(mat)) / std if std > 0 else mat - np.mean(mat)
    elif transform == 'minmax':
        mn, mx = np.min(mat), np.max(mat)
        mat = (mat - mn) / (mx - mn) if mx > mn else np.zeros_like(mat)
    elif transform is not None:
        raise ValueError(
            f"Unknown transform '{transform}': expected one of None, "
            f"'log1p', 'sqrt', 'zscore', 'minmax'."
        )

    if save_path is not None:
        fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)

        im = ax.imshow(
            mat,
            cmap="Reds",
            interpolation="nearest",
            origin="lower",
            aspect="equal",
            rasterized=True,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Contact frequency", fontsize=14)

        ax.set_title("Average Contact Map", fontsize=18, pad=12)
        ax.set_xlabel("Genomic bin", fontsize=14)
        ax.set_ylabel("Genomic bin", fontsize=14)

        ax.tick_params(axis="both", labelsize=12)

        if save:
            plt.savefig(save_path, format="svg", dpi=500, bbox_inches="tight")
            np.save(save_path.replace(".svg", ".npy"), mat)

        plt.close(fig)

    return mat

def heats_to_prob(heats,path,burnin,q=0.15):
    q_dist = np.quantile(np.array(heats),1-q)
    prob_mat = np.zeros(heats[0].shape)

    norm = np.zeros(len(heats[0]))
    for heat in heats:
        for i in range(len(heats[0])):
            norm[i]+=(np.average(np.diagonal(heat,offset=i))+np.average(np.diagonal(heat,offset=-i)))/2
    norm = norm/len(heats)

    for i in range(burnin,len(heats)):
        prob_mat[heats[i]>q_dist] += 1
    
    prob_mat = prob_mat/len(heats)
    for i in range(len(prob_mat)):
        for j in range(0,len(prob_mat)-i):
            prob_mat[i,i+j]=prob_mat[i,i+j]/norm[j]
            prob_mat[i+j,i]=prob_mat[i+j,i]/norm[j]
    
    figure(figsize=(10, 10))
    plt.imshow(prob_mat,cmap="Reds")
    plt.colorbar()
    plt.title(f'Normalized Probability distribution that distance < {q} quantile', fontsize=13)
    plt.savefig(path,format='png',dpi=500)
    plt.show(block=False)

def binned_distance_matrix(idx,folder_name,input=None,th=23):
    '''
    This function calculates the mean distance through models, between two specific beads.
    We do that for all the possible beads and we take a matrix/heatmap.
    This one may take some hours for many beads or models.
    This works for .pdb files.
    '''
    
    V = get_coordinates_pdb(folder_name+f'/pdbs/SM{idx}.pdb')
    mat = distance.cdist(V, V, 'euclidean') # this is the way \--/ 

    figure(figsize=(25, 20))
    plt.imshow(mat,cmap=LinearSegmentedColormap.from_list("bright_red",[(1,0,0),(1,1,1)]), vmin=0, vmax=th)
    # plt.colorbar();
    # plt.title('Binned Distance heatmap',fontsize=16)
    plt.savefig(folder_name+f'/heatmaps/SM_bindist_heatmap_idx{idx}.png',format='png',dpi=500)
    plt.close()

    np.save(folder_name+f'/heatmaps/binned_dist_matrix_idx{idx}.npy',mat)
    
    return mat

def average_binned_distance_matrix(folder_name,N_steps,step,burnin,th1=0,th2=23):
    '''
    This function calculates the mean distance through models, between two specific beads.
    We do that for all the possible beads and we take a matrix/heatmap.
    This one may take some hours for many beads or models.
    smoothing (str): You can choose between 'Nearest Neighbour', 'bilinear', 'hanning', 'bicubic'.
    '''
    sum_mat = 0
    for i in tqdm(range(0,N_steps,step)):
        V = get_coordinates_pdb(folder_name+f'/pdbs/SM{i}.pdb')
        if i >= burnin*step:
            sum_mat += distance.cdist(V, V, 'euclidean') # this is the way \--/ 
    new_N = N_steps//step
    avg_mat = sum_mat/new_N
    
    figure(figsize=(25, 20))
    plt.imshow(avg_mat,cmap=LinearSegmentedColormap.from_list("bright_red",[(1,0,0),(1,1,1)]), vmin=th1, vmax=th2)
    # plt.colorbar();
    # plt.title('Average Binned Distance heatmap',fontsize=16)
    plt.savefig(folder_name+f'/plots/SM_avg_bindist_heatmap.png',format='png',dpi=500)
    plt.show(block=False)
    np.save(folder_name+'/plots/average_binned_dist_matrix.npy',avg_mat)

    return avg_mat

########## Statistics ###########
def get_stats(ms,ns,N_beads):
    '''
    This is a function that computes maximum compaction score in every step of the simulation.
    '''
    # Computing Folding Metrics
    N_coh = len(ms)
    chromatin = np.zeros(N_beads)
    chromatin2 = np.zeros(N_beads)
    for nn in range(N_coh):
        m,n = int(ms[nn]),int(ns[nn])
        if m<=n:
            chromatin[m:n] = 1
            chromatin2[m:n]+=1
        else:
            chromatin[0:n] = 1
            chromatin[m:] = 1
            chromatin2[0:n]+=1
            chromatin2[m:]+=1
    f = np.mean(chromatin)
    F = np.mean(chromatin2)
    f_std = np.std(chromatin)
    FC = 1/(1-f+0.001)
    
    return f, f_std, F, FC

def count_parents_children(ms,ns,N_beads):
    '''
    This function counts how many child and parent loops we have on the system.
    '''
    # Computing the folding vector
    N_coh = len(ms)
    chromatin = np.zeros(N_beads)
    for nn in range(N_coh):
        m,n = int(ms[nn]),int(ns[nn])
        if m<=n:
            chromatin[m:n]+=1
        else:
            chromatin[0:n]+=1
            chromatin[m:]+=1

    # Compute number of parents and children.
    N_parents, N_children = 0, 0
    for nn in range(N_coh):
        m, n = int(ms[nn]), int(ns[nn])
        if len(np.unique(chromatin[m:n]))==1:
            N_children+=1
        elif len(np.unique(chromatin[m:n]))>1:
            N_parents+=1

    return N_parents, N_children

def angle3d(x, y):
    '''
    By Krzystof Banecki.
    '''
    norm1 = np.linalg.norm(x)
    norm2 = np.linalg.norm(y)
    if norm1==0 or norm2==0:
        return np.pi
    cosine3d = sum(x*y)/norm1/norm2
    if cosine3d<=-1:
        return np.pi
    if cosine3d>=1:
        return 0
    return np.arccos(cosine3d)

def total_angle3d(structure):
    '''
    By Krzystof Banecki.
    '''
    return sum([angle3d(np.array(structure[i-1])-np.array(structure[i]),
                        np.array(structure[i+1])-np.array(structure[i]))
                for i in range(1, len(structure)-1)])/np.pi/(len(structure)-2)

def save_info(N_beads,N_coh,N_CTCF,kappa,f,b,avg_loop,path,N_steps,MC_step,burnin,mode,ufs,Es,Ks,Fs,Bs):
    file = open(path+'/other/info.txt', "w")
    file.write(f'Number of beads {N_beads}\n')
    file.write(f'Number of cohesins {N_coh}\n')
    file.write(f'Number of CTCFs {N_CTCF}\n')
    file.write(f'Average loop size {avg_loop}\n')
    file.write(f'f = {f}, b={b}, k={kappa}\n')
    file.write(f'Monte Carlo parameters: N_steps={N_steps}, MC_step={MC_step}, burnin={burnin*MC_step}, method {mode}\n')
    file.write(f'Equillibrium parameters: uf={np.average(ufs)}, E={np.average(Es)}, K={np.average(Ks)}, F={np.average(Fs)}, B={np.average(Bs)}')
