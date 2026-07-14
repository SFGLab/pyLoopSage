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
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
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
 
SIM_COLOR = "#2563eb"   # blue  - simulated
EXP_COLOR = "#dc2626"   # red   - experimental
NEUTRAL_COLOR = "#374151"
 
def _apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.6,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
 
def _save_fig(fig, path, plot_name):
    fig.savefig(os.path.join(path, 'plots', f'{plot_name}.png'), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(path, 'plots', f'{plot_name}.pdf'), bbox_inches="tight")
    plt.close(fig)
 
# ============================================================================
# Format detection & loading
# ============================================================================
 
def detect_interaction_format(interaction_file):
    """Auto-detect .bedpe / .bed / .narrowPeak from the file extension."""
    ext = os.path.splitext(interaction_file)[-1].lower()
    if ext == '.bedpe':
        return 'bedpe'
    elif ext == '.bed':
        return 'bed'
    elif ext == '.narrowpeak':
        return 'narrowpeak'
    else:
        raise ValueError(
            f"Unsupported interaction file format '{ext}' for {interaction_file}. "
            f"Expected .bedpe, .bed, or .narrowPeak."
        )
 
def _load_interaction_df(interaction_file, region, chrom, paired):
    """Read and region/chrom-filter the interaction file. `paired=True` also
    filters on the second anchor's coordinates (columns 4/5), used for .bedpe."""
    df = pd.read_csv(interaction_file, sep='\t', header=None, comment='#')
 
    mask = (df[0] == chrom) & (df[1] >= region[0]) & (df[2] >= region[0]) & \
           (df[1] < region[1]) & (df[2] < region[1])
 
    if paired:
        mask &= (df[4] >= region[0]) & (df[5] >= region[0]) & \
                (df[4] < region[1]) & (df[5] < region[1])
 
    return df[mask].reset_index(drop=True)
 
def _get_weight_column(df, col_idx, default=1.0):
    """Numeric weight column, falling back to `default` for missing/non-numeric
    values (e.g. a '.' score field) or a missing column altogether."""
    if col_idx < df.shape[1]:
        return pd.to_numeric(df[col_idx], errors='coerce').fillna(default).values
    return np.full(len(df), default)
 
def _compute_bead_positions(starts, ends, region, resolution, N_beads):
    mid = (starts.values + ends.values) // 2
    beads = ((mid - region[0]) // resolution).astype(int)
    return np.clip(beads, 0, N_beads - 1)
 
def build_bedpe_pairs(df, region, resolution, N_beads, weight_col=6):
    """(x, y, weight) triplets - one per loop, x/y are the anchor bead indices."""
    x = _compute_bead_positions(df[1], df[2], region, resolution, N_beads)
    y = _compute_bead_positions(df[4], df[5], region, resolution, N_beads)
    w = _get_weight_column(df, weight_col)
    return list(zip(x, y, w))
 
def build_single_region_sites(df, region, resolution, N_beads, weight_col=4):
    """(x, weight) pairs - one per CTCF site, for .bed / .narrowPeak input."""
    x = _compute_bead_positions(df[1], df[2], region, resolution, N_beads)
    w = _get_weight_column(df, weight_col)
    return list(zip(x, w))
 
# ============================================================================
# Signal construction (loop/site strength projected onto beads)
# ============================================================================
 
def build_strength_signal(mat_sim, interaction_file, fmt, region, chrom, N_beads):
    """
    Build the experimental vs. simulated "strength" vectors used for the
    original loop/site-strength correlation.
 
    - .bedpe: each loop's PET count is added to both anchor beads (exp_vec),
      and mat_sim[x, y] is added to both anchor beads (th_vec) - same logic
      as the original bedpe-only function.
    - .bed / .narrowPeak: each CTCF site only has one bead, so its score is
      added there (exp_vec), and the simulated signal (th_vec) is that bead's
      total row contact strength in mat_sim, since there's no second anchor
      to look up a specific mat_sim[x, y] entry.
    """
    resolution = max((region[1] - region[0]) // N_beads, 1)
    exp_vec = np.zeros(N_beads)
    th_vec = np.zeros(N_beads)
 
    if fmt == 'bedpe':
        df = _load_interaction_df(interaction_file, region, chrom, paired=True)
        for x, y, w in build_bedpe_pairs(df, region, resolution, N_beads):
            exp_vec[x] += w
            exp_vec[y] += w
            th_vec[x] += mat_sim[x, y]
            th_vec[y] += mat_sim[x, y]
    else:
        df = _load_interaction_df(interaction_file, region, chrom, paired=False)
        for x, w in build_single_region_sites(df, region, resolution, N_beads):
            exp_vec[x] += w
            th_vec[x] += mat_sim[x, :].sum() - mat_sim[x, x]
 
    return exp_vec, th_vec
 
# ============================================================================
# Matrix-derived signals (P(s) decay, insulation score, compartment eigenvector)
# ============================================================================
 
def compute_ps_curve(mat):
    """Average contact strength vs. genomic distance (diagonal decay)."""
    N = mat.shape[0]
    ps = np.array([np.mean(np.diagonal(mat, offset=k)) for k in range(N)])
    return ps
 
def compute_ps_curve_from_bedpe(interaction_file, region, chrom, N_beads):
    """
    Approximate an experimental P(s) decay curve directly from loop PET
    counts binned by anchor distance - no full contact matrix needed, but
    only possible for .bedpe (paired-anchor) input.
    """
    resolution = max((region[1] - region[0]) // N_beads, 1)
    df = _load_interaction_df(interaction_file, region, chrom, paired=True)
 
    ps_sum = np.zeros(N_beads)
    ps_count = np.zeros(N_beads)
    for x, y, w in build_bedpe_pairs(df, region, resolution, N_beads):
        d = abs(y - x)
        ps_sum[d] += w
        ps_count[d] += 1
 
    return np.divide(ps_sum, ps_count, out=np.zeros_like(ps_sum), where=ps_count > 0)
 
def compute_insulation_score(mat, window=5):
    """Diamond insulation score: mean contact strength in a window x window
    square straddling each bead's diagonal position."""
    N = mat.shape[0]
    ins = np.zeros(N)
    for i in range(window, N - window):
        ins[i] = np.mean(mat[i - window:i, i + 1:i + window + 1])
    return ins
 
def compute_compartment_eigenvector(mat):
    """First eigenvector (PC1) of the bead-bead correlation matrix - the
    standard A/B compartment signal."""
    mat = np.nan_to_num(mat)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(mat)
    corr = np.nan_to_num(corr)
    eigvals, eigvecs = np.linalg.eigh(corr)
    return eigvecs[:, -1]  # eigh returns ascending eigenvalues; last = largest
 
# ============================================================================
# Correlation reporting (logging + file + plot), shared by every metric
# ============================================================================
 
def _plot_signal_comparison(exp_vec, th_vec, pears, spear, signal_name, path, plot_name, xlabel):
    """Three-panel figure: experimental trace, simulated trace, and a
    regression scatter of the two - more informative than two bare line
    plots, and the scatter makes the correlation strength visually obvious."""
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
 
    ax_scatter.scatter(exp_vec, th_vec, s=20, color=NEUTRAL_COLOR, alpha=0.55,
                        edgecolor="white", linewidth=0.3)
    if len(exp_vec) > 1 and np.std(exp_vec) > 0:
        coeffs = np.polyfit(exp_vec, th_vec, 1)
        xs = np.linspace(np.min(exp_vec), np.max(exp_vec), 100)
        ax_scatter.plot(xs, np.polyval(coeffs, xs), color="#111111", lw=2, linestyle="--")
    ax_scatter.set_xlabel("Experimental")
    ax_scatter.set_ylabel("Simulated")
    ax_scatter.set_title("Correlation", fontsize=13)
    ax_scatter.text(
        0.05, 0.95, f"Pearson r = {pears:.3f}\nSpearman ρ = {spear:.3f}",
        transform=ax_scatter.transAxes, va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d1d5db")
    )
 
    _save_fig(fig, path, plot_name)
 
def report_correlation(exp_vec, th_vec, label, signal_name, f, path=None,
                        plot_name=None, xlabel="Genomic distance (simulation beads)"):
    """Compute Pearson/Spearman/Kendall correlation between two 1D signals,
    log + write it, and optionally save a comparison plot (see
    _plot_signal_comparison). Returns the Pearson coefficient (or None if
    there wasn't enough data)."""
    exp_vec, th_vec = np.asarray(exp_vec), np.asarray(th_vec)
 
    if len(exp_vec) < 2 or len(th_vec) < 2 or len(exp_vec) != len(th_vec):
        log.warning(f"Not enough/mismatched data to compute {label} correlation "
                    f"for {signal_name} - skipping.")
        return None
 
    pears, pval1 = pearsonr(th_vec, exp_vec)
    spear, pval2 = spearmanr(th_vec, exp_vec)
    kendal, pval3 = kendalltau(th_vec, exp_vec)
 
    log.info(f'-------------- {label} Correlation ({signal_name}) -----------')
    log.info(f'Pearson Correlation: {pears:.3f} with pvalue {pval1}.')
    log.info(f'Spearman Correlation: {spear:.3f} with pvalue {pval2}.')
    log.info(f'Kendall Correlation: {kendal:.3f} with pvalue {pval3}.\n')
 
    f.write(f'---- {label} Estimations ({signal_name}) ----\n')
    f.write(f'Pearson Correlation: {pears:.3f} with pvalue {pval1}.\n')
    f.write(f'Spearman Correlation: {spear:.3f} with pvalue {pval2}.\n')
    f.write(f'Kendall Correlation: {kendal:.3f} with pvalue {pval3}.\n\n')
 
    if plot_name and path is not None:
        _plot_signal_comparison(exp_vec, th_vec, pears, spear, signal_name, path, plot_name, xlabel)
 
    return pears
 
# ============================================================================
# Heatmap-vs-heatmap comparison plots (need a full experimental matrix)
# ============================================================================
 
def plot_heatmaps_side_by_side(mat_sim, mat_exp, path, plot_name="heatmaps_side_by_side",
                                cmap="Reds", log_scale=True):
    """Experimental and simulated contact maps side by side on a shared color scale."""
    sim = np.log1p(mat_sim) if log_scale else mat_sim
    exp = np.log1p(mat_exp) if log_scale else mat_exp
    vmax = np.nanpercentile(np.concatenate([sim.ravel(), exp.ravel()]), 99)
 
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    axs[0].imshow(exp, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
    axs[0].set_title("Experimental", fontsize=14, fontweight="bold")
    im1 = axs[1].imshow(sim, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
    axs[1].set_title("Simulated", fontsize=14, fontweight="bold")
 
    for ax in axs:
        ax.set_xlabel("Bead index")
        ax.grid(False)
    axs[0].set_ylabel("Bead index")
 
    fig.colorbar(im1, ax=axs, shrink=0.8, label="log(1 + contacts)" if log_scale else "Contacts")
    fig.suptitle("Simulated vs. Experimental Contact Maps", fontsize=16, fontweight="bold")
 
    _save_fig(fig, path, plot_name)
 
def plot_matrix_triangle_comparison(mat_sim, mat_exp, path, plot_name="matrix_triangle_comparison",
                                     cmap="Reds", log_scale=True, pears=None):
    """Classic Hi-C-style split heatmap: experimental in the upper triangle,
    simulated in the lower triangle of a single square - the most direct way
    to eyeball agreement between the two."""
    N = mat_sim.shape[0]
    sim = np.log1p(mat_sim) if log_scale else mat_sim.copy()
    exp = np.log1p(mat_exp) if log_scale else mat_exp.copy()
 
    combined = np.zeros_like(sim, dtype=np.float64)
    iu = np.triu_indices(N, k=1)
    il = np.tril_indices(N, k=-1)
    combined[iu] = exp[iu]
    combined[il] = sim[il]
    np.fill_diagonal(combined, np.nan)
 
    vmax = np.nanpercentile(combined, 99)
 
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(combined, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
    ax.plot([0, N - 1], [0, N - 1], color="black", lw=1, alpha=0.6)
    ax.grid(False)
 
    ax.text(0.98, 0.03, "Simulated", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, fontweight="bold", color="#111111")
    ax.text(0.02, 0.97, "Experimental", transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color="#111111")
 
    title = "Simulated vs. Experimental (triangle split)"
    if pears is not None:
        title += f"  •  Pearson r = {pears:.3f}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Bead index")
    ax.set_ylabel("Bead index")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log(1 + contacts)" if log_scale else "Contacts")
 
    _save_fig(fig, path, plot_name)
 
def plot_difference_heatmap(mat_sim, mat_exp, path, plot_name="difference_heatmap", cmap="RdBu_r"):
    """Symmetric log2-ratio map: red = simulated-enriched, blue = experimental-enriched,
    white = agreement - useful for spotting where the model over/under-predicts contacts."""
    eps = 1e-6
    ratio = np.log2((mat_sim + eps) / (mat_exp + eps))
    vmax = np.nanpercentile(np.abs(ratio), 99)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ratio, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
    ax.grid(False)
    ax.set_title("Simulated vs. Experimental (log\u2082 ratio)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Bead index")
    ax.set_ylabel("Bead index")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log\u2082(simulated / experimental)")
 
    _save_fig(fig, path, plot_name)
 
# ============================================================================
# Main entry point
# ============================================================================
 
def corr_exp_heat(mat_sim, interaction_file, region, chrom, N_beads, path,
                   mat_exp=None, insulation_window=5):
    """
    Correlate the simulated contact matrix against experimental signal(s).
 
    `interaction_file` format (.bedpe / .bed / .narrowPeak) is auto-detected
    from its extension, and the loop/site-strength signal is built
    accordingly (see build_strength_signal()). This correlation is always
    computed, exactly like the original function.
 
    Three additional correlations are attempted:
    - P(s) diagonal decay: derived directly from .bedpe loop distances when
      available, otherwise needs `mat_exp`.
    - Insulation score and compartment eigenvector: both need a genuine
      experimental contact matrix - they can't be derived from loop/peak
      calls alone - so they only run if `mat_exp` (an N_beads x N_beads
      experimental matrix, e.g. from a .cool/.hic file at the same
      resolution) is supplied. Otherwise they're skipped with a log message
      explaining why, rather than being silently omitted.
 
    Returns
    -------
    pears : float
        Pearson correlation of the (always-computed) loop/site-strength signal.
    """
    fmt = detect_interaction_format(interaction_file)
    log.info(f"Detected interaction file format: {fmt}")
 
    _apply_plot_style()
    os.makedirs(os.path.join(path, 'other'), exist_ok=True)
    os.makedirs(os.path.join(path, 'plots'), exist_ok=True)
    corr_path = os.path.join(path, 'other', 'correlations.txt')
 
    with open(corr_path, "w") as f:
        # ---- 1. Loop / site strength (always available) ----
        exp_vec, th_vec = build_strength_signal(mat_sim, interaction_file, fmt, region, chrom, N_beads)
 
        pears = report_correlation(
            exp_vec, th_vec, label="Optimistic", signal_name="loop/site strength",
            f=f, path=path, plot_name="pearson"
        )
 
        # Fixed vs. the original: use ONE combined mask so exp_vec/th_vec stay
        # aligned, instead of masking each vector by the other's zero positions.
        nonzero = (exp_vec != 0) & (th_vec != 0)
        report_correlation(
            exp_vec[nonzero], th_vec[nonzero], label="Pessimistic", signal_name="loop/site strength",
            f=f, path=path, plot_name=None
        )
 
        # ---- 2. P(s) diagonal decay ----
        sim_ps = compute_ps_curve(mat_sim)
        if fmt == 'bedpe':
            exp_ps = compute_ps_curve_from_bedpe(interaction_file, region, chrom, N_beads)
            report_correlation(
                exp_ps, sim_ps, label="P(s) decay", signal_name="diagonal decay",
                f=f, path=path, plot_name="ps_decay",
                xlabel="Genomic distance (simulation beads)"
            )
        elif mat_exp is not None:
            exp_ps = compute_ps_curve(mat_exp)
            report_correlation(
                exp_ps, sim_ps, label="P(s) decay", signal_name="diagonal decay",
                f=f, path=path, plot_name="ps_decay",
                xlabel="Genomic distance (simulation beads)"
            )
        else:
            log.info("Skipping P(s) diagonal-decay correlation: only possible directly "
                     "from .bedpe loop distances, or by passing an experimental "
                     "contact matrix via `mat_exp`.")
 
        # ---- 3. Insulation score & compartment eigenvector ----
        if mat_exp is not None:
            ins_sim = compute_insulation_score(mat_sim, insulation_window)
            ins_exp = compute_insulation_score(mat_exp, insulation_window)
            report_correlation(
                ins_exp, ins_sim, label="Insulation score", signal_name="insulation score",
                f=f, path=path, plot_name="insulation"
            )
 
            eig_sim = compute_compartment_eigenvector(mat_sim)
            eig_exp = compute_compartment_eigenvector(mat_exp)
            report_correlation(
                eig_exp, eig_sim, label="Eigenvector", signal_name="compartment eigenvector (PC1)",
                f=f, path=path, plot_name="eigenvector"
            )
        else:
            log.info("Skipping insulation-score and eigenvector correlations: pass an "
                     "experimental contact matrix via `mat_exp` to enable these - they "
                     "can't be derived from loop/peak calls alone.")
 
        # ---- 4. Heatmap-vs-heatmap comparison plots ----
        if mat_exp is not None:
            plot_heatmaps_side_by_side(mat_sim, mat_exp, path)
            plot_matrix_triangle_comparison(mat_sim, mat_exp, path, pears=pears)
            plot_difference_heatmap(mat_sim, mat_exp, path)
        else:
            log.info("Skipping heatmap-vs-heatmap comparison plots: pass an experimental "
                     "contact matrix via `mat_exp` to enable these.")
 
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

def get_heatmap(mm_vec,save_path=None,th=1,save=False):
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
    H (np.array): a heatmap of the 3D structure.
    '''
    V = get_coordinates_mm(mm_vec)
    mat = distance.cdist(V, V, 'euclidean') # this is the way \--/
    mat = 1/(mat+1)

    if save_path!=None:
        figure(figsize=(25, 20))
        plt.imshow(mat,cmap="Reds")
        if save: plt.savefig(save_path,format='svg',dpi=500)
        plt.close()
        if save: np.save(save_path.replace("svg", "npy"),mat)
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
