import numpy as np
import pyBigWig
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from matplotlib.pyplot import figure

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
        J_mode="binary",
        J_norm=None,
        alpha=1.0,
        smooth=False,
        smooth_sigma=2.0
    ):
    '''
    Construct chromatin interaction structures from BEDPE loop data.

    Returns
    -------
    L : (N_beads,)
        Left binding profile.

    R : (N_beads,)
        Right binding profile.

    J : (N_beads, N_beads)
        Interaction (adjacency) matrix.
    
    J_mode
    -------
    binary   : strict graph adjacency (0/1 only, no weights)
    strength : weighted by BEDPE score
    distance : exponential decay with loop size
    '''

    if region is None:
        if chrom not in CHROM_LENGTHS:
            raise ValueError(f"Unknown chromosome: {chrom}")

        region = [0, CHROM_LENGTHS[chrom]]

    # --------------------------------------------------
    # Validate inputs
    # --------------------------------------------------
    if J_mode not in ["binary", "strength", "distance"]:
        raise ValueError(f"Invalid J_mode: {J_mode}")

    if J_norm not in [None, "global", "row"]:
        raise ValueError(f"Invalid J_norm: {J_norm}")

    if N_beads <= 10:
        raise ValueError("N_beads too small")

    if region[1] <= region[0]:
        raise ValueError("Invalid region: end must be > start")

    if alpha <= 0 and J_mode == "distance":
        raise ValueError("alpha must be positive for distance mode")

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

    J = np.zeros((N_beads, N_beads), dtype=np.float64)
    L = np.zeros(N_beads, dtype=np.float64)
    R = np.zeros(N_beads, dtype=np.float64)

    distances = []

    # --------------------------------------------------
    # Build L, R, J
    # --------------------------------------------------
    for i in range(len(df)):

        x = (df[1][i] + df[2][i]) // 2
        y = (df[4][i] + df[5][i]) // 2

        x = min(max(x, 0), N_beads - 1)
        y = min(max(y, 0), N_beads - 1)

        distances.append(np.abs(y - x))

        # ==================================================
        # J construction
        # ==================================================
        if J_mode == "binary":
            J[x, y] = 1.0
            J[y, x] = 1.0

        else:
            if J_mode == "strength":
                weight = float(df[6][i])

            elif J_mode == "distance":
                weight = np.exp(-alpha * np.abs(y - x))

            else:
                weight = 1.0

            for a in range(x, y):
                for b in range(x, y):
                    J[a, b] += weight

        # ==================================================
        # L / R (unchanged logic)
        # ==================================================
        if has_col_7_8:
            if df[7][i] >= 0:
                L[x] += df[6][i] * (1 - df[7][i])
                R[x] += df[6][i] * df[7][i]

            if df[8][i] >= 0:
                L[y] += df[6][i] * (1 - df[8][i])
                R[y] += df[6][i] * df[8][i]
        else:
            L[x] += df[6][i]
            L[y] += df[6][i]
            R[x] += df[6][i]
            R[y] += df[6][i]

    # --------------------------------------------------
    # Backbone (ONLY for non-binary or as structural prior)
    # --------------------------------------------------
    if diagonal_interactions:
        if J_mode != "binary":
            for i in range(N_beads - 1):
                J[i, i + 1] += 1.0
                J[i + 1, i] += 1.0
        else:
            for i in range(N_beads - 1):
                J[i, i + 1] = 1.0
                J[i + 1, i] = 1.0

    # --------------------------------------------------
    # Normalization (disabled for binary)
    # --------------------------------------------------
    if J_mode != "binary":

        if J_norm == "global":
            s = np.sum(J)
            if s > 0:
                J /= s

        elif J_norm == "row":
            for i in range(N_beads):
                s = np.sum(J[i])
                if s > 0:
                    J[i] /= s

    # --------------------------------------------------
    # Optional smoothing
    # --------------------------------------------------
    if smooth:
        from scipy.ndimage import gaussian_filter1d
        L = gaussian_filter1d(L, sigma=smooth_sigma)
        R = gaussian_filter1d(R, sigma=smooth_sigma)

    # --------------------------------------------------
    # L / R normalization
    # --------------------------------------------------
    if normalization:
        L = L / (np.sum(L) + 1e-12)
        R = R / (np.sum(R) + 1e-12)

    # --------------------------------------------------
    # STATISTICS (NEW ADDITION)
    # --------------------------------------------------

    distances_arr = np.array(distances)

    # only real loop edges (exclude backbone)
    loop_edges = np.argwhere(J > 0)

    # remove backbone neighbors
    loop_edges = np.array([
        (i, j) for (i, j) in loop_edges
        if np.abs(i - j) > 1
    ])

    if len(loop_edges) > 0:
        loop_lengths = np.abs(loop_edges[:, 1] - loop_edges[:, 0])
    else:
        loop_lengths = np.array([])

    statistics = {
        "n_loops": int(len(loop_lengths)),
        "distance_between_loops": {
            "mean": float(np.mean(distances_arr)) if len(distances_arr) else 0.0,
            "median": float(np.median(distances_arr)) if len(distances_arr) else 0.0,
            "min": float(np.min(distances_arr)) if len(distances_arr) else 0.0,
            "max": float(np.max(distances_arr)) if len(distances_arr) else 0.0,
        },
        "loop_length": {
            "mean": float(np.mean(loop_lengths)) if len(loop_lengths) else 0.0,
            "median": float(np.median(loop_lengths)) if len(loop_lengths) else 0.0,
            "min": float(np.min(loop_lengths)) if len(loop_lengths) else 0.0,
            "max": float(np.max(loop_lengths)) if len(loop_lengths) else 0.0,
        }
    }

    if viz:
        print("\n" + "=" * 60)
        print("              LOOP STATISTICS (beads units)")
        print("=" * 60)

        print(f"\nNumber of loops (non-backbone): {statistics['n_loops']}\n")

        d = statistics["distance_between_loops"]
        print("Distance between loop anchors:")
        print(f"  mean   : {d['mean']:.3f}")
        print(f"  median : {d['median']:.3f}")
        print(f"  min    : {d['min']:.3f}")
        print(f"  max    : {d['max']:.3f}")

        l = statistics["loop_length"]
        print("\nLoop length:")
        print(f"  mean   : {l['mean']:.3f}")
        print(f"  median : {l['median']:.3f}")
        print(f"  min    : {l['min']:.3f}")
        print(f"  max    : {l['max']:.3f}")

        print("\n" + "=" * 60 + "\n")

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    if viz:

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].plot(L, label="L", color="green")
        axs[0].plot(R, label="R", color="red")
        axs[0].set_title("Binding profiles")
        axs[0].legend()
        axs[0].grid()

        sns.histplot(distances, bins=10, ax=axs[1])
        axs[1].set_title("Loop size distribution")
        axs[1].grid()

        if out_path is not None:
            save_path = out_path + "/plots/LR_data.svg"
            plt.savefig(save_path, format="svg", dpi=600)
            save_path = out_path + "/plots/LR_data.png"
            plt.savefig(save_path, format="png", dpi=600)
            save_path = out_path + "/plots/LR_data.pdf"
            plt.savefig(save_path, format="pdf", dpi=600)

        plt.tight_layout()
        plt.close()

        # --------------------------------------------------
        # Heatmap (separate figure)
        # --------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        im = ax.imshow(J, cmap="viridis", origin="lower")
        ax.set_title("Adjacency matrix J")
        ax.set_xlabel("bead i")
        ax.set_ylabel("bead j")

        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        if out_path is not None:
            save_path = out_path + "/plots/J_data.svg"
            plt.savefig(save_path, format="svg", dpi=600)
            save_path = out_path + "/plots/J_data.png"
            plt.savefig(save_path, format="png", dpi=600)
            save_path = out_path + "/plots/J_data.pdf"
            plt.savefig(save_path, format="pdf", dpi=600)
        plt.close()


        # --------------------------------------------------
        # Graph diagnostics (strength + distribution)
        # --------------------------------------------------
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        strength = np.sum(J, axis=0)

        axs[0].plot(strength, color="black")
        axs[0].set_title("Node strength")
        axs[0].set_xlabel("bead index")
        axs[0].set_ylabel("strength")
        axs[0].grid()

        axs[1].hist(strength, bins=10, color="steelblue", edgecolor="black")
        axs[1].set_title("Strength distribution")
        axs[1].set_xlabel("strength")
        axs[1].set_ylabel("count")
        axs[1].grid()

        plt.tight_layout()
        plt.close()
    
    return L, R, J, statistics

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

        if region is None:
            if chrom not in CHROM_LENGTHS:
                raise ValueError(f"Unknown chromosome: {chrom}")

            region = [0, CHROM_LENGTHS[chrom]]

    # --------------------------------------------------
    # Core loader
    # --------------------------------------------------
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

        # --------------------------------------------------
        # normalization stage
        # --------------------------------------------------
        if norm is not None:
            weights = self.normalize(weights, method=norm)

        # --------------------------------------------------
        # optional [-1,1] scaling
        # --------------------------------------------------
        if scale_minus1_1:
            xmin, xmax = np.min(weights), np.max(weights)
            if xmax > xmin:
                weights = 2.0 * (weights - xmin) / (xmax - xmin) - 1.0
            else:
                weights = weights * 0.0

        # --------------------------------------------------
        # VISUALIZATION (centralized here)
        # --------------------------------------------------
        if viz:

            x = np.arange(len(weights))

            fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            # --------------------------
            # 1. main signal
            # --------------------------
            axs[0].plot(x, weights, color="black", lw=1.5)
            axs[0].axhline(0, color="red", ls="--", lw=1)
            axs[0].set_title("BigWig signal (bead-resolved)")
            axs[0].set_ylabel("signal")
            axs[0].grid(alpha=0.3)

            # --------------------------
            # 2. histogram (distribution)
            # --------------------------
            axs[1].hist(weights, bins=40, color="steelblue", edgecolor="black")
            axs[1].set_title("Signal distribution")
            axs[1].set_ylabel("count")
            axs[1].grid(alpha=0.3)

            # --------------------------
            # 3. running smooth view (structure intuition)
            # --------------------------
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

    # --------------------------------------------------
    # Load FULL genome once (IMPORTANT FIX)
    # --------------------------------------------------
    df_full = pd.read_csv(bed_file, sep="\t", header=None)

    # region-specific view (used for signal only)
    df = df_full[
        (df_full[0] == chrom) &
        (df_full[1] < region[1]) &
        (df_full[2] > region[0])
    ].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No compartments found in region")

    # --------------------------------------------------
    # GLOBAL normalization reference (CRITICAL FIX)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # build signal
    # --------------------------------------------------
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

    # --------------------------------------------------
    # fill signal (UNCHANGED LOGIC)
    # --------------------------------------------------
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
                print(f"[DEBUG] label fallback: {df[3][i]} -> {val:.3f}")

        signal[b0:b1 + 1] += val
        counts[b0:b1 + 1] += 1.0

    # normalize coverage
    mask = counts > 0
    signal[mask] /= counts[mask]

    # --------------------------------------------------
    # smoothing (UNCHANGED)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # GLOBAL normalization FIX (THIS IS THE IMPORTANT CHANGE)
    # --------------------------------------------------
    if scale_minus1_1:
        if gmax > gmin:
            signal = 2.0 * (signal - gmin) / (gmax - gmin) - 1.0
        else:
            signal[:] = 0.0

    # --------------------------------------------------
    # visualization (UNCHANGED)
    # --------------------------------------------------
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

    # ------------------------------------------------------------
    # STEP 1: load BEDPE and extract chr1 loops
    # ------------------------------------------------------------
    df = pd.read_csv(bedpe_file, sep="\t", header=None)
    df = df[df[0] == chrom].reset_index(drop=True)

    starts = np.minimum(df[1].values, df[4].values)
    ends   = np.maximum(df[2].values, df[5].values)

    # ------------------------------------------------------------
    # STEP 2: define "TAD-like" candidate regions
    # (simple heuristic: cluster loops by midpoints)
    # ------------------------------------------------------------
    midpoints = (starts + ends) // 2

    # sort and pick dense region window
    sorted_idx = np.argsort(midpoints)
    midpoints_sorted = midpoints[sorted_idx]

    window_size = 200  # number of loops per region

    # pick random window
    i0 = np.random.randint(0, len(midpoints_sorted) - window_size)
    sel = midpoints_sorted[i0:i0 + window_size]

    region_start = int(np.min(sel)) - 200_000
    region_end   = int(np.max(sel)) + 200_000

    # clip to chr bounds
    region_start = max(region_start, chr_start)
    region_end   = min(region_end, chr_end)

    region = [region_start, region_end]

    print("Selected region:", region)

    # ------------------------------------------------------------
    # STEP 3: call your function
    # ------------------------------------------------------------
    N_beads = 200

    L, R, J, _ = binding_vectors_from_bedpe(
        bedpe_file=bedpe_file,
        N_beads=N_beads,
        region=region,
        chrom=chrom,
        normalization=True,
        viz=True,          # we will do custom viz below
        J_mode="binary",
        J_norm="global",
        alpha=0.01,
        smooth=True,
        smooth_sigma=2.0
    )

    # ------------------------------------------------------------
    # STEP 5: Compartments (BW)
    # ------------------------------------------------------------

    print("========= Load Compartments ============")

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
    print("========= Load Compartments (BED) ==========")

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