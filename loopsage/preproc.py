import numpy as np
import pyBigWig
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from matplotlib.pyplot import figure

def binding_vectors_from_bedpe(
        bedpe_file, N_beads, region, chrom,
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
            # STRICT adjacency: no weights, no accumulation
            J[x, y] = 1.0
            J[y, x] = 1.0

        else:
            if J_mode == "strength":
                weight = float(df[6][i])

            elif J_mode == "distance":
                weight = np.exp(-alpha * np.abs(y - x))

            else:
                weight = 1.0

            # accumulate weighted block
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
    # Visualization
    # --------------------------------------------------
    if viz:

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].plot(L, label="L", color="green")
        axs[0].plot(R, label="R", color="red")
        axs[0].set_title("Binding profiles")
        axs[0].legend()
        axs[0].grid()

        sns.histplot(distances, bins=80, ax=axs[1])
        axs[1].set_title("Loop size distribution")
        axs[1].grid()

        plt.tight_layout()
        plt.show()

        # --------------------------------------------------
        # Graph view of J
        # --------------------------------------------------
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        im = axs[0].imshow(J, cmap="viridis", origin="lower")
        axs[0].set_title("Adjacency matrix J")
        plt.colorbar(im, ax=axs[0], fraction=0.046)

        strength = np.sum(J, axis=0)

        axs[1].plot(strength, color="black")
        axs[1].set_title("Node strength")
        axs[1].grid()

        axs[2].hist(strength, bins=50, color="steelblue", edgecolor="black")
        axs[2].set_title("Strength distribution")
        axs[2].grid()

        plt.tight_layout()
        plt.show()

    return L, R, J

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

def load_track(file,region,chrom,N_beads,viz=False,roll=False):
    bw = pyBigWig.open(file)
    weights = bw_to_array(bw, region, chrom, N_beads,viz,roll)
    return weights[:N_beads]

def bw_to_array(bw, region, chrom, N_beads, viz=False, roll=False):
    step = (region[1]-region[0])//N_beads
    bw_array = bw.values(chrom, region[0], region[1])
    bw_array = np.nan_to_num(bw_array)
    bw_array_new = list()
    for i in range(step,len(bw_array)+1,step):
        bw_array_new.append(np.average(bw_array[(i-step):i]))
    weights = (np.roll(np.array(bw_array_new),3)+np.roll(np.array(bw_array_new),-3))/2 if roll else bw_array_new
    if viz:
        figure(figsize=(15, 5))
        plt.plot(weights)
        plt.grid()
        plt.title('ChIP-Seq signal',fontsize=20)
        plt.close()
    
    return weights

def main():
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

    L, R, J = binding_vectors_from_bedpe(
        bedpe_file=bedpe_file,
        N_beads=N_beads,
        region=region,
        chrom=chrom,
        normalization=True,
        viz=True,          # we will do custom viz below
        J_mode="strength",
        J_norm="global",
        alpha=0.01,
        smooth=True,
        smooth_sigma=2.0
    )

if __name__ == "__main__":
    main()