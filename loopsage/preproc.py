import os
import numpy as np
import pyBigWig
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from .logger import get_logger

log = get_logger(__name__)

CHROM_LENGTHS = {
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
    "chrM": 16_569
}

def binding_vectors_from_bedpe(
        bedpe_file, N_beads, chrom, region=None, out_path=None,
        normalization=False,
        viz=False,
        diagonal_interactions=True,
        alpha=1.0,
        smooth=False,
        smooth_sigma=2.0,
        contrastive=True
    ):
    '''
    Construct chromatin interaction structures from BEDPE loop data.

    Returns
    -------
    L : (N_beads,)
    R : (N_beads,)
    J : (N_beads, N_beads)        -> discrete adjacency
    J_loss : (N_beads, N_beads)   -> smoothed continuous energy landscape
    statistics : dict
    '''
    try:
        region = [int(region[0]), int(region[1])]
        if region[1] <= region[0]:
            raise ValueError("Invalid region bounds")

    except Exception:
        log.warning("Invalid region provided → using full chromosome.")
        if chrom not in CHROM_LENGTHS:
            raise ValueError(f"Unknown chromosome: {chrom}")
        region = [0, CHROM_LENGTHS[chrom]]

    df = pd.read_csv(bedpe_file, sep='\t', header=None)

    df = df[
        (df[1] >= region[0]) & (df[2] >= region[0]) &
        (df[4] >= region[0]) & (df[5] >= region[0]) &
        (df[5] < region[1]) & (df[4] < region[1]) &
        (df[1] < region[1]) & (df[2] < region[1]) &
        (df[0] == chrom)
    ].reset_index(drop=True)

    resolution = (region[1] - region[0]) // N_beads

    df[1], df[2], df[4], df[5] = (
        (df[1] - region[0]) // resolution,
        (df[2] - region[0]) // resolution,
        (df[4] - region[0]) // resolution,
        (df[5] - region[0]) // resolution
    )

    has_col_7_8 = df.shape[1] > 8

    if has_col_7_8:
        p7 = df[7].values
        p8 = df[8].values

        def valid_prob_vector(p):
            p = np.asarray(p)
            finite = np.isfinite(p)
            allowed = (p == -1) | ((p >= 0) & (p <= 1))
            return np.all(finite & allowed)

        valid_p7 = valid_prob_vector(p7)
        valid_p8 = valid_prob_vector(p8)

        if not (valid_p7 and valid_p8):
            log.warning("Columns 7/8 detected but invalid probabilities → ignoring them.")
            has_col_7_8 = False

    # --------------------------------------------------
    # Initialize
    # --------------------------------------------------
    J = np.zeros((N_beads, N_beads), dtype=np.float64)
    J_loss = np.zeros((N_beads, N_beads), dtype=np.float64)

    L = np.zeros(N_beads, dtype=np.float64)
    R = np.zeros(N_beads, dtype=np.float64)

    distances = []
    centers = []   # ✅ NEW

    eps = 1e-12

    # --------------------------------------------------
    # Build adjacency + weighted contact map
    # --------------------------------------------------
    for i in range(len(df)):

        x = (df[1][i] + df[2][i]) // 2
        y = (df[4][i] + df[5][i]) // 2

        x = min(max(x, 0), N_beads - 1)
        y = min(max(y, 0), N_beads - 1)

        if x == y:
            continue

        dist = abs(y - x)
        distances.append(dist)

        # ✅ NEW: store center
        centers.append((x + y) // 2)

        # binary adjacency
        J[x, y] = 1.0
        J[y, x] = 1.0

        w = df[6][i]

        if has_col_7_8:
            p7 = df[7][i]
            p8 = df[8][i]

            if p7 >= 0:
                wx = w * (1 - p7)
                wy = w * p7
            else:
                wx = wy = w * 0.5

            if p8 >= 0:
                wx2 = w * (1 - p8)
                wy2 = w * p8
            else:
                wx2 = wy2 = w * 0.5

            J_loss[x, y] += 0.5 * (wx + wy2)
            J_loss[y, x] += 0.5 * (wy + wx2)

            L[x] += wx
            R[x] += wy
            L[y] += wx2
            R[y] += wy2

        else:
            J_loss[x, y] += w
            J_loss[y, x] += w

            L[x] += w
            L[y] += w
            R[x] += w
            R[y] += w

    # Backbone
    if diagonal_interactions:
        for i in range(N_beads - 1):
            J[i, i + 1] = 1.0
            J[i + 1, i] = 1.0

    J_loss = (J_loss.T + J_loss) / 2

    if smooth:
        L = gaussian_filter1d(L, sigma=smooth_sigma, mode="nearest")
        R = gaussian_filter1d(R, sigma=smooth_sigma, mode="nearest")
        J_loss = gaussian_filter(J_loss.astype(np.float64), sigma=smooth_sigma)
    else:
        J_loss = J_loss.astype(np.float64)

    L, R = L / np.mean(L), R / np.mean(R)

    if contrastive:
        eps = 1e-6

        J_loss = J_loss / (np.max(J_loss) + eps)
        J_loss = (J_loss - np.mean(J_loss)) / (np.std(J_loss) + eps)
        J_loss = np.tanh(J_loss)

        L = np.tanh((L - np.mean(L)) / (np.std(L) + eps))
        R = np.tanh((R - np.mean(R)) / (np.std(R) + eps))

    # --------------------------------------------------
    # STATISTICS (FIXED PART)
    # --------------------------------------------------
    distances_arr = np.array(distances)

    loop_edges = np.argwhere(J > 0)
    loop_edges = np.array([
        (i, j) for (i, j) in loop_edges if np.abs(i - j) > 1
    ])

    loop_lengths = np.abs(loop_edges[:, 1] - loop_edges[:, 0]) if len(loop_edges) else np.array([])

    # ✅ NEW: inter-loop distances
    centers = np.array(centers)
    if len(centers) > 1:
        inter_loop_distances = np.abs(np.diff(np.sort(centers)))
    else:
        inter_loop_distances = np.array([])

    statistics = {
        "n_loops": int(len(loop_lengths)),
        "distance_between_loops": {
            "mean": float(np.mean(inter_loop_distances)) if len(inter_loop_distances) else 0.0,
            "median": float(np.median(inter_loop_distances)) if len(inter_loop_distances) else 0.0,
            "min": float(np.min(inter_loop_distances)) if len(inter_loop_distances) else 0.0,
            "max": float(np.max(inter_loop_distances)) if len(inter_loop_distances) else 0.0,
        },
        "loop_length": {
            "mean": float(np.mean(loop_lengths)) if len(loop_lengths) else 0.0,
            "median": float(np.median(loop_lengths)) if len(loop_lengths) else 0.0,
            "min": float(np.min(loop_lengths)) if len(loop_lengths) else 0.0,
            "max": float(np.max(loop_lengths)) if len(loop_lengths) else 0.0,
        }
    }

    # units
    unit_beads = "beads"
    unit_bp = f"{resolution} bp"

    log.info("🧬 Chromatin Loop Stats")
    log.info("=" * 40)

    log.info(f"Total loops: {statistics['n_loops']}")

    # -------------------------------------------------
    # Distance between loop anchors
    # -------------------------------------------------
    d = statistics["distance_between_loops"]

    log.info("📏 Distance between loop anchors:")
    log.info(
        f"mean   : {d['mean']:.2f} {unit_beads}  "
        f"(~{d['mean'] * resolution:.0f} bp)"
    )
    log.info(
        f"median : {d['median']:.2f} {unit_beads}  "
        f"(~{d['median'] * resolution:.0f} bp)"
    )
    log.info(
        f"min    : {d['min']:.2f} {unit_beads}  "
        f"(~{d['min'] * resolution:.0f} bp)"
    )
    log.info(
        f"max    : {d['max']:.2f} {unit_beads}  "
        f"(~{d['max'] * resolution:.0f} bp)"
    )

    # -------------------------------------------------
    # Loop lengths
    # -------------------------------------------------
    l = statistics["loop_length"]

    log.info("🔗 Loop lengths:")
    log.info(
        f"mean   : {l['mean']:.2f} {unit_beads}  "
        f"(~{l['mean'] * resolution:.0f} bp)"
    )
    log.info(
        f"median : {l['median']:.2f} {unit_beads}  "
        f"(~{l['median'] * resolution:.0f} bp)"
    )
    log.info(
        f"min    : {l['min']:.2f} {unit_beads}  "
        f"(~{l['min'] * resolution:.0f} bp)"
    )
    log.info(
        f"max    : {l['max']:.2f} {unit_beads}  "
        f"(~{l['max'] * resolution:.0f} bp)"
    )

    log.info("=" * 40)

    # VISUALIZATION
    if viz:
        if out_path is not None:
            plot_dir = os.path.join(out_path, "plots")
            os.makedirs(plot_dir, exist_ok=True)

        # ---- L/R
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

        # Left panel: L and R
        axs[0].plot(L, label="L (left binding)", lw=2, color="darkgreen")
        axs[0].plot(R, label="R (right binding)", lw=2, color="darkred")

        axs[0].set_title("Binding profiles along chromatin")
        axs[0].set_xlabel("Bead index")
        axs[0].set_ylabel("Signal strength")

        axs[0].legend(frameon=False)
        axs[0].grid(alpha=0.3)
        
        # Right panel: loop stats
        axs[1].hist(distances, bins=20, color="steelblue", edgecolor="black", alpha=0.8)

        axs[1].set_title("Loop length distribution")
        axs[1].set_xlabel("Distance (beads)")
        axs[1].set_ylabel("Frequency")
        axs[1].grid(alpha=0.3)

        plt.suptitle("Chromatin loop organization summary", fontsize=14)
        plt.tight_layout()

        # save
        if out_path is not None:
            plt.savefig(os.path.join(plot_dir, "LR_profiles.svg"), dpi=600, format="svg")
            plt.savefig(os.path.join(plot_dir, "LR_profiles.png"), dpi=600, format="png")
            plt.savefig(os.path.join(plot_dir, "LR_profiles.pdf"), dpi=600, format="pdf")

        plt.close()

        J_plot = (J - np.mean(J)) / (np.std(J) + 1e-8)
        plt.figure(figsize=(6, 5))
        plt.imshow(
            J_plot,
            cmap="RdBu_r",
            origin="lower",
            vmin=-2, vmax=2
        )
        plt.title("Discrete J (z-score normalized)")
        plt.colorbar()

        if out_path is not None:
            plt.savefig(os.path.join(plot_dir, "J_discrete.svg"), dpi=600, format="svg")
            plt.savefig(os.path.join(plot_dir, "J_discrete.png"), dpi=600, format="png")
            plt.savefig(os.path.join(plot_dir, "J_discrete.pdf"), dpi=600, format="pdf")

        plt.close()

        # ---- J_loss heatmap
        plt.figure(figsize=(6, 5))
        plt.imshow(J_loss, cmap="magma", origin="lower")
        plt.title("Smoothed J_loss")
        plt.colorbar()

        if out_path is not None:
            plt.savefig(os.path.join(plot_dir, "J_loss.svg"), dpi=600, format="svg")
            plt.savefig(os.path.join(plot_dir, "J_loss.png"), dpi=600, format="png")
            plt.savefig(os.path.join(plot_dir, "J_loss.pdf"), dpi=600, format="pdf")

        plt.close()

        # ---- 3D surface plot of loss landscape
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        x = np.arange(N_beads)
        X, Y = np.meshgrid(x, x)

        ax.plot_surface(X, Y, -J_loss, cmap="magma", linewidth=0)

        ax.set_title("Energy landscape (J_loss)")
        ax.set_xlabel("i")
        ax.set_ylabel("j")

        # --------------------------------------------------
        # VIEW FROM TOP (2D-like projection of surface)
        # --------------------------------------------------
        ax.view_init(elev=-30, azim=60)

        plt.tight_layout()

        if out_path is not None:
            plt.savefig(os.path.join(plot_dir, "J_loss_surface.svg"), dpi=600, format="svg")
            plt.savefig(os.path.join(plot_dir, "J_loss_surface.png"), dpi=600, format="png")
            plt.savefig(os.path.join(plot_dir, "J_loss_surface.pdf"), dpi=600, format="pdf")

        plt.close()

    return 2*L, 2*R, J, np.sqrt(len(loop_lengths))*J_loss, statistics

def get_rnap_energy(path,region,chrom,N_beads,normalization):
    '''
    For the RNApII potential.

    Input:
    path (str): path with bw file that determines RNApII binding.
    region (list): a list with two integers [start,end], which represent the start and end point of the region of interest.
    chrom (str): chromosome of interest.
    normalization (bool): in case that it is needed to normalize to numpy arrays that represent RNApII binding potential.
    '''
    signal = load_track(path,region,chrom,N_beads)
    if normalization: signal = signal/np.sum(signal)
    return signal

def distance_point_line(x0,y0,a=1,b=-1,c=0):
    return np.abs(a*x0+b*y0+c)/np.sqrt(a**2+b**2)

class BWExporter:
    """
    Lightweight exporter for genomic signal tracks from BigWig files.

    Supports:
    - ChIP-seq
    - compartments
    - any continuous genomic signal
    """

    def __init__(self, path, N_beads, chrom, region=None):
        """
        Core configuration is now stored inside the object.
        """
        self.file = path
        self.region = region
        self.chrom = chrom
        self.N_beads = N_beads

        try:
            region = [int(region[0]), int(region[1])]

            # sanity check
            if region[1] <= region[0]:
                raise ValueError("Invalid region bounds")

        except Exception:
            log.warning("Invalid region provided → using full chromosome.")

            if chrom not in CHROM_LENGTHS:
                raise ValueError(f"Unknown chromosome: {chrom}")

            region = [0, CHROM_LENGTHS[chrom]]

    # Core loader
    def load_track(self,
                   viz=False,
                   roll=False,
                   norm=None,
                   out_path=None,
                   scale_minus1_1=False):
        """
        Load BigWig track and convert to bead resolution.

        Parameters
        ----------
        norm:
            None        -> raw
            "log"       -> log(1 + x)
            "zscore"    -> standardize
            "minmax"    -> [0,1]

        scale_minus1_1:
            if True -> map signal from [0,1] to [-1,1]
        """

        bw = pyBigWig.open(self.file)
        weights = self.bw_to_array(
            bw,
            self.region,
            self.chrom,
            self.N_beads,
            viz=False,
            roll=roll
        )
        bw.close()

        weights = np.array(weights[:self.N_beads], dtype=np.float64)

        # normalization stage
        if norm is not None:
            weights = self.normalize(weights, method=norm)

        # optional [-1,1] scaling
        if scale_minus1_1:
            xmin, xmax = np.min(weights), np.max(weights)
            if xmax > xmin:
                weights = 2.0 * (weights - xmin) / (xmax - xmin) - 1.0
            else:
                weights = weights * 0.0

        # VISUALIZATION (centralized here)
        if viz:

            x = np.arange(len(weights))

            fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            # 1. main signal
            axs[0].plot(x, weights, color="black", lw=1.5)
            axs[0].axhline(0, color="red", ls="--", lw=1)
            axs[0].set_title("BigWig signal (bead-resolved)")
            axs[0].set_ylabel("signal")
            axs[0].grid(alpha=0.3)

            # 2. histogram (distribution)
            axs[1].hist(weights, bins=40, color="steelblue", edgecolor="black")
            axs[1].set_title("Signal distribution")
            axs[1].set_ylabel("count")
            axs[1].grid(alpha=0.3)

            # 3. running smooth view (structure intuition)
            smooth = (np.roll(weights, 1) + weights + np.roll(weights, -1)) / 3

            axs[2].plot(x, smooth, color="darkgreen", lw=1.5)
            axs[2].set_title("Smoothed signal (local structure)")
            axs[2].set_xlabel("bead index")
            axs[2].set_ylabel("signal")
            axs[2].grid(alpha=0.3)

            if out_path is not None:
                save_path = out_path + "/plots/bw_dataset.svg"
                plt.savefig(save_path, format="svg", dpi=600)
                save_path = out_path + "/plots/bw_dataset.png"
                plt.savefig(save_path, format="png", dpi=600)
                save_path = out_path + "/plots/bw_dataset.pdf"
                plt.savefig(save_path, format="pdf", dpi=600)

            plt.tight_layout()
            plt.close()

        return weights

    # --------------------------------------------------
    # BigWig -> bead array
    # --------------------------------------------------
    def bw_to_array(self, bw, region, chrom, N_beads,
                    viz=False, roll=False):

        step = (region[1] - region[0]) // N_beads

        raw = bw.values(chrom, region[0], region[1])
        raw = np.nan_to_num(raw)

        binned = []
        for i in range(step, len(raw) + 1, step):
            binned.append(np.mean(raw[i - step:i]))

        weights = np.array(binned, dtype=np.float64)

        if roll:
            weights = (np.roll(weights, 3) + np.roll(weights, -3)) / 2

        return weights

    def compute_global_minmax(self):
        """
        Computes genome-wide min/max from BigWig.
        This is the KEY FIX for consistent normalization.
        """

        bw = pyBigWig.open(self.file)

        values = bw.values(self.chrom, 0, CHROM_LENGTHS[self.chrom])
        values = np.nan_to_num(values)

        bw.close()

        self.global_minmax = (np.min(values), np.max(values))

    # --------------------------------------------------
    # Normalization utilities
    # --------------------------------------------------
    def normalize(self, x, method="log"):

        if method == "log":
            return np.log1p(x)

        elif method == "zscore":
            std = np.std(x)
            if std == 0:
                return x - np.mean(x)
            return (x - np.mean(x)) / std

        elif method == "minmax":
            xmin, xmax = self.global_minmax()
            if xmax == xmin:
                return x * 0
            return (x - xmin) / (xmax - xmin)

        else:
            raise ValueError(f"Unknown normalization: {method}")

def load_compartments_bed(
    bed_file,
    region,
    chrom,
    N_beads,
    out_path=None,
    use_score=True,
    spline_smooth=False,
    spline_s=1.0,
    scale_minus1_1=True,
    viz=False,
    debug=False
):
    """
    Load compartment BED file and convert to bead-level signal.
    """

    try:
        region = [int(region[0]), int(region[1])]

        # sanity check
        if region[1] <= region[0]:
            raise ValueError("Invalid region bounds")

    except Exception:
        log.warning("Invalid region provided → using full chromosome.")

        if chrom not in CHROM_LENGTHS:
            raise ValueError(f"Unknown chromosome: {chrom}")

        region = [0, CHROM_LENGTHS[chrom]]

    # Load FULL genome once (IMPORTANT FIX)
    df_full = pd.read_csv(bed_file, sep="\t", header=None)

    # region-specific view (used for signal only)
    df = df_full[
        (df_full[0] == chrom) &
        (df_full[1] < region[1]) &
        (df_full[2] > region[0])
    ].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No compartments found in region")

    # GLOBAL normalization reference (CRITICAL FIX)
    def extract_val(row):
        if use_score:
            try:
                return 2.0 * float(row[4]) - 1.0
            except:
                pass

        # fallback label
        label = row[3]
        if isinstance(label, str):
            if label.startswith("A"):
                sign = +1.0
            elif label.startswith("B"):
                sign = -1.0
            else:
                return 0.0
            depth = len(label.split("."))
            return sign * (1.0 + 0.2 * (depth - 1))

        return 0.0

    # compute global min/max ON FULL DATASET
    global_vals = np.array([extract_val(df_full.iloc[i]) for i in range(len(df_full))])

    gmin, gmax = np.min(global_vals), np.max(global_vals)

    # build signal
    signal = np.zeros(N_beads, dtype=np.float64)
    counts = np.zeros(N_beads, dtype=np.float64)

    resolution = (region[1] - region[0]) // N_beads

    def parse_label(label):
        if not isinstance(label, str):
            return 0.0
        if label.startswith("A"):
            sign = +1.0
        elif label.startswith("B"):
            sign = -1.0
        else:
            return 0.0
        depth = len(label.split("."))
        return sign * (1.0 + 0.2 * (depth - 1))

    # fill signal (UNCHANGED LOGIC)
    for i in range(len(df)):

        start = max(df[1][i], region[0])
        end   = min(df[2][i], region[1])

        b0 = int((start - region[0]) // resolution)
        b1 = int((end   - region[0]) // resolution)

        b0 = max(0, min(N_beads - 1, b0))
        b1 = max(0, min(N_beads - 1, b1))

        val = None

        if use_score:
            try:
                val = 2.0 * float(df[4][i]) - 1.0
            except:
                val = None

        if val is None:
            val = parse_label(df[3][i])
            if debug:
                log.warning(f"label fallback: {df[3][i]} -> {val:.3f}")

        signal[b0:b1 + 1] += val
        counts[b0:b1 + 1] += 1.0

    # normalize coverage
    mask = counts > 0
    signal[mask] /= counts[mask]

    # smoothing (UNCHANGED)
    if spline_smooth:
        from scipy.interpolate import UnivariateSpline

        x = np.arange(N_beads)
        valid = mask & np.isfinite(signal)

        if np.sum(valid) > 3:
            signal_filled = signal.copy()
            signal_filled[~valid] = np.interp(
                x[~valid], x[valid], signal[valid]
            )

            spline = UnivariateSpline(x, signal_filled, s=spline_s)
            signal = spline(x)

    # GLOBAL normalization FIX (THIS IS THE IMPORTANT CHANGE)
    if scale_minus1_1:
        if gmax > gmin:
            signal = 2.0 * (signal - gmin) / (gmax - gmin) - 1.0
        else:
            signal[:] = 0.0

    # visualization (UNCHANGED)
    if viz:

        import matplotlib.pyplot as plt

        x = np.arange(N_beads)

        plt.figure(figsize=(14, 4))
        plt.plot(x, signal, color="black", lw=1.5)

        plt.fill_between(x, 0, signal, where=(signal > 0),
                         color="red", alpha=0.5)

        plt.fill_between(x, 0, signal, where=(signal < 0),
                         color="blue", alpha=0.5)

        plt.axhline(0, color="black", ls="--", lw=1)

        plt.title("Compartment signal (global normalization)")
        plt.xlabel("bead index")
        plt.ylabel("signal [-1,1]")
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if out_path is not None:
            plt.savefig(out_path + "/plots/comparments.svg", format="svg", dpi=600)

        plt.close()

    return signal

def main():
    print("========= Loop Preprocessing ==========")
    # ------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------
    bedpe_file = "/home/blackpianocat/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe"

    chrom = "chr1"

    # chr1 approximate bounds in your dataset
    chr_start = 0
    chr_end   = 249_000_000  # human chr1 length approx

    # STEP 1: load BEDPE and extract chr1 loops
    df = pd.read_csv(bedpe_file, sep="\t", header=None)
    df = df[df[0] == chrom].reset_index(drop=True)

    starts = np.minimum(df[1].values, df[4].values)
    ends   = np.maximum(df[2].values, df[5].values)

    # STEP 2: define "TAD-like" candidate regions
    # (simple heuristic: cluster loops by midpoints)
    midpoints = (starts + ends) // 2

    # sort and pick dense region window
    sorted_idx = np.argsort(midpoints)
    midpoints_sorted = midpoints[sorted_idx]

    window_size = 200  # number of loops per region

    # pick random window
    i0 = np.random.randint(0, len(midpoints_sorted) - window_size)
    sel = midpoints_sorted[i0:i0 + window_size]

    region_start = int(np.min(sel)) - 50_000
    region_end   = int(np.max(sel)) + 50_000

    # clip to chr bounds
    region_start = max(region_start, chr_start)
    region_end   = min(region_end, chr_end)

    region = [region_start, region_end]

    print("Selected region:", region)

    # STEP 3: call your function
    N_beads = 200

    L, R, J, J_loss, _ = binding_vectors_from_bedpe(
        bedpe_file=bedpe_file,
        N_beads=N_beads,
        region=region,
        chrom=chrom,
        normalization=False,
        viz=True,
        alpha=0.01,
        smooth=True,
        smooth_sigma=N_beads/50
    )

    # STEP 4: Compartments (BW)
    log.info("========= Load Compartments ============")

    bw_file = "/home/blackpianocat/Data/ENCODE/ENCSR968KAY_HiC/ENCFF412CDH_comps.bigWig"
    chrom = "chr1"
    region = region
    N_beads = 200

    # INIT exporter (now stores everything inside)
    exporter = BWExporter(
        path=bw_file,
        region=region,
        chrom=chrom,
        N_beads=N_beads
    )
    
    # LOAD COMPARTMENT SIGNAL
    comps = exporter.load_track(
        viz=True,
        roll=True,
        norm=None,        # "log", "minmax", None
        scale_minus1_1=True  # optional, default behavior
    )

    # ------------------------------------------------------------
    # STEP 5: Compartments (BED)
    # ------------------------------------------------------------
    log.info("========= Load Compartments (BED) ==========")

    bed_comp_file = "/home/blackpianocat/Data/Trios/HiChIP/HiChIP_Subcompartments/calder_subc_50k/bed/dsGM19238_30.bed"

    comps_bed = load_compartments_bed(
        bed_file=bed_comp_file,
        region=region,
        chrom=chrom,
        N_beads=N_beads,
        use_score=True,
        spline_smooth=True,     # NEW
        spline_s=1.0,            # NEW (smoothing strength)
        scale_minus1_1=True,
        viz=True,
        debug=True             # useful to see fallback behavior
    )

if __name__ == "__main__":
    main()