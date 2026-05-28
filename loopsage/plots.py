import imageio
import shutil
import os
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
from matplotlib.pyplot import cm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats
from tqdm import tqdm
from scipy import stats

METHODS = ("binary", "loop_fill", "gaussian", "tanh")
 
_META = {
    "binary":    ("Binary\n(anchor contacts only)",           "Oranges"),
    "loop_fill": ("Loop-fill\n(uniform interior)",            "Blues"),
    "gaussian":  ("Gaussian-weighted\n(decay from axis)",     "Greens"),
    "tanh":      ("Tanh-sigmoid\n(smooth boundary roll-off)", "Purples"),
}

def make_loop_hist(Ms,Ns,path=None):
    Ls = np.abs(Ns-Ms).flatten()
    Ls_df = pd.DataFrame(Ls)
    figure(figsize=(10, 7), dpi=600)
    sns.histplot(data=Ls_df, bins=30,  kde=True,stat='density')
    plt.grid()
    plt.legend()
    plt.ylabel('Probability',fontsize=16)
    plt.xlabel('Loop Length',fontsize=16)
    if path!=None:
        save_path = path+'/plots/loop_length.png'
        plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    Is, Js = Ms.flatten(), Ns.flatten()
    IJ_df = pd.DataFrame()
    IJ_df['mi'] = Is
    IJ_df['nj'] = Js
    figure(figsize=(8, 8), dpi=600)
    sns.jointplot(IJ_df, x="mi", y="nj",kind='hex',color='Red')
    if path!=None:
        save_path = path+'/plots/ij_prob.png'
        plt.savefig(save_path,format='png',dpi=200)
    plt.close()

def make_gif(N,path=None):
    with imageio.get_writer('plots/arc_video.gif', mode='I') as writer:
        for i in range(N):
            image = imageio.imread(f"plots/arcplots/arcplot_{i}.png")
            writer.append_data(image)
    save_path = path+"/plots/arcplots/" if path!=None else "/plots/arcplots/"
    shutil.rmtree(save_path)

def make_timeplots(Es, Bs, Ks, Fs, burnin, mode, path=None):
    figure(figsize=(10, 8), dpi=600)
    plt.plot(Es, 'k')
    plt.plot(Bs, 'cyan')
    plt.plot(Ks, 'green')
    plt.plot(Fs, 'red')
    plt.axvline(x=burnin, color='blue')
    plt.ylabel('Metrics', fontsize=16)
    plt.ylim((np.min(Es)-10,-np.min(Es)))
    plt.xlabel('Monte Carlo Step', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(['Total Energy', 'Binding', 'crossing', 'Folding'], fontsize=16)
    plt.grid()

    if path!=None:
        save_path = path+'/plots/energies.png'
        plt.savefig(save_path,format='png',dpi=200)
    plt.close()

    # Autocorrelation plot
    if mode=='Annealing':
        x = np.arange(0,len(Fs[burnin:])) 
        p3 = np.poly1d(np.polyfit(x, Fs[burnin:], 3))
        ys = np.array(Fs)[burnin:]-p3(x)
    else:
        ys = np.array(Fs)[burnin:]
    plot_acf(ys, title=None, lags = len(np.array(Fs)[burnin:])//2)
    plt.ylabel("Autocorrelations", fontsize=16)
    plt.xlabel("Lags", fontsize=16)
    plt.grid()
    if path!=None: 
        save_path = path+'/plots/autoc.png'
        plt.savefig(save_path,dpi=200)
    plt.close()

def make_moveplots(unbinds, slides, path=None):
    figure(figsize=(10, 8), dpi=600)
    plt.plot(unbinds, 'blue')
    plt.plot(slides, 'red')
    plt.ylabel('Number of moves', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(['Rebinding', 'Sliding'], fontsize=16)
    plt.grid()
    if path!=None:
        save_path = path+'/plots/moveplot.png'
        plt.savefig(save_path,dpi=200)
    plt.close()

def average_pooling(mat,dim_new):
    im = Image.fromarray(mat)
    size = dim_new,dim_new
    im_resized = np.array(im.resize(size))
    return im_resized

def plot_epi_trajectory(epi_states, path=None, cmap="bwr"):
    """
    Plot epigenetic state trajectory as a heatmap.

    Parameters
    ----------
    epi_states : array (N_beads, N_steps)
        Epigenetic states over time

    path : str or None
        Output directory (optional)

    vmin, vmax : float
        Color scale limits (default assumes [-1, 1])

    cmap : str
        Colormap (default: blue-white-red)
    """

    # Ensure numpy array (important for consistency)
    epi_states = np.asarray(epi_states, dtype=np.float64)
    vmin = np.min(epi_states)
    vmax = np.max(epi_states)

    # --------------------------------------------------
    # Figure (publication style)
    # --------------------------------------------------
    plt.figure(figsize=(10, 10), dpi=200)

    im = plt.imshow(
        epi_states,
        aspect="auto",        # time stretched nicely
        origin="lower",       # bead 0 at bottom
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------
    plt.xlabel("Monte Carlo step")
    plt.ylabel("Bead index")
    plt.title("Epigenetic trajectory")

    # --------------------------------------------------
    # Colorbar
    # --------------------------------------------------
    cbar = plt.colorbar(im)
    cbar.set_label("Epigenetic signal")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if path is not None:
        save_path = path + "/plots/epigenetic_trajectory.svg"
        plt.savefig(save_path, format="svg", dpi=600)
        save_path = path + "/plots/epigenetic_trajectory.png"
        plt.savefig(save_path, format="png", dpi=600)
        save_path = path + "/plots/epigenetic_trajectory.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)

    plt.tight_layout()
    plt.close()

def coh_traj_plot(ms, ns, N_beads, path, jump_threshold=400, min_stable_time=3):
    """
    Plots the trajectories of cohesins as filled regions between their two ends over time.

    Parameters:
        ms (list of arrays): List where each element is an array of left-end positions of a cohesin over time.
        ns (list of arrays): List where each element is an array of right-end positions of a cohesin over time.
        N_beads (int): Total number of beads (simulation sites) in the system.
        path (str): Directory path where the plots will be saved.
        jump_threshold (int, optional): Maximum allowed jump (in bead units) between consecutive time points for both ends.
            If the jump between two consecutive positions exceeds this threshold for either end, that segment is considered a jump and is masked out.
            Lower values make the criterion for erasing (masking) trajectories more strict (more segments are erased), higher values make it less strict.
        min_stable_time (int, optional): Minimum number of consecutive time points required for a region to be considered stable and shown.
            Shorter stable regions (less than this value) are erased (masked out).
            Higher values make the criterion more strict (only longer stable regions are shown), lower values make it less strict.

    The function highlights only stable, non-jumping regions of cohesin trajectories.
    """
    print('\nPlotting trajectories of cohesins...')
    N_coh = len(ms)
    figure(figsize=(10, 10), dpi=200)
    cmap = plt.get_cmap('prism')
    colors = [cmap(i / N_coh) for i in range(N_coh)]

    for nn in tqdm(range(N_coh)):
        tr_m, tr_n = np.array(ms[nn]), np.array(ns[nn])
        steps = np.arange(len(tr_m))

        # Calculate jump size for tr_m and tr_n independently
        jumps_m = np.abs(np.diff(tr_m))
        jumps_n = np.abs(np.diff(tr_n))

        # Create mask: True = good point, False = jump
        jump_mask = np.ones_like(tr_m, dtype=bool)
        jump_mask[1:] = (jumps_m < jump_threshold) & (jumps_n < jump_threshold)  # both must be below threshold

        # Now we want to detect stable regions
        stable_mask = np.copy(jump_mask)

        # Find connected regions
        current_length = 0
        for i in range(len(stable_mask)):
            if jump_mask[i]:
                current_length += 1
            else:
                if current_length < min_stable_time:
                    stable_mask[i-current_length:i] = False
                current_length = 0
        # Handle last region
        if current_length < min_stable_time:
            stable_mask[len(stable_mask)-current_length:] = False

        # Apply mask
        tr_m_masked = np.ma.masked_array(tr_m, mask=~stable_mask)
        tr_n_masked = np.ma.masked_array(tr_n, mask=~stable_mask)

        plt.fill_between(steps, tr_m_masked, tr_n_masked,
                         color=colors[nn], alpha=0.6, interpolate=False, linewidth=0)
    plt.xlabel('MC Step', fontsize=16)
    plt.ylabel('Simulation Beads', fontsize=16)
    plt.gca().invert_yaxis()
    plt.ylim((0, N_beads))
    save_path = path + '/plots/LEFs.png'
    plt.savefig(save_path, format='png',dpi=200)
    plt.close()

def coh_probdist_plot(ms,ns,N_beads,path):
    Ntime = len(ms[0,:])
    M = np.zeros((N_beads,Ntime))
    for ti in range(Ntime):
        m,n = ms[:,ti], ns[:,ti]
        M[m,ti]+=1
        M[n,ti]+=1
    dist = np.average(M,axis=1)

    figure(figsize=(15, 6), dpi=600)
    x = np.arange(N_beads)
    plt.fill_between(x,dist)
    plt.title('Probablity distribution of cohesin')
    save_path = path+'/plots/coh_probdist.png' if path!=None else 'coh_trajectories.png'
    plt.savefig(save_path, format='png', dpi=200)
    plt.close()

def stochastic_heatmap(ms, ns, L, path, method="gaussian", viz=True):
    """
    Compute an averaged Hi-C heatmap from cohesin (LEF) trajectories.
 
    All four methods are fully vectorised over N_lef × T using NumPy
    broadcasting — no Python loops over time steps or LEFs.
 
    Parameters
    ----------
    ms : np.ndarray, shape (N_lef, T)
        Left-anchor bead positions over time (0-indexed, values in [0, L-1]).
    ns : np.ndarray, shape (N_lef, T)
        Right-anchor bead positions over time (0-indexed, values in [0, L-1]).
    L : int
        Number of polymer beads (output matrix is L × L).
    path : str
        Root output directory; SVG saved to <path>/plots/stochastic_heatmap.svg.
    method : str
        Reconstruction method: "binary" | "loop_fill" | "gaussian" | "tanh" | "all".
        Default: "gaussian".
    viz : bool
        If True (default) plot and save the heatmap SVG. If False, skip plotting.
 
    Returns
    -------
    dict[str, np.ndarray]
        {method_name: contact_matrix (L, L)}.
 
    Method descriptions
    -------------------
    binary    — accumulates +1 only at the two anchor positions (i0, i1) per LEF
                per snapshot; produces a sparse map of cohesin anchor co-localisation.
 
    loop_fill — every bead pair inside [i0, i1] × [i0, i1] receives +1, modelling
                uniform 3-D proximity of the extruded loop; gives TAD-like blocks.
 
    gaussian  — a 2-D Gaussian blob centred at the loop midpoint, width σ = loop/3,
                mimics polymer contact-probability decay; smooth diagonal stripes.
 
    tanh      — per-bead weight is tanh((b-i0+0.5)/s)·tanh((i1-b+0.5)/s) with
                s = max(loop×0.05, 1); flat plateau inside the loop, sigmoidal edges.
    """
    assert ms.shape == ns.shape, "ms and ns must have the same shape (N_lef, T)"
 
    method = method.lower()
    if method not in METHODS and method != "all":
        raise ValueError(f"method must be one of {('all',) + METHODS}, got '{method}'")
 
    N_lef, T = ms.shape
    left  = np.clip(np.minimum(ms, ns).astype(np.int32), 0, L - 1)  # (N_lef, T)
    right = np.clip(np.maximum(ms, ns).astype(np.int32), 0, L - 1)  # (N_lef, T)
 
    # Flat view: work with all N_lef*T pairs at once
    l_flat = left.ravel()   # (N_lef*T,)
    r_flat = right.ravel()  # (N_lef*T,)
    S      = l_flat.size    # total number of (LEF, snapshot) pairs
    b      = np.arange(L, dtype=np.float32)  # bead indices (L,)
 
    selected = list(METHODS) if method == "all" else [method]
    results  = {}
 
    for name in selected:
        mat = np.zeros((L, L), dtype=np.float64)
 
        # ---- chunk size: balance memory vs progress-bar granularity ----
        chunk = max(1, min(S, 4096))
        n_chunks = (S + chunk - 1) // chunk
 
        with tqdm(total=S, desc=f"{name:10s}", unit="LEF·t",
                  dynamic_ncols=True, colour="green") as pbar:
 
            for start in range(0, S, chunk):
                sl  = slice(start, start + chunk)
                l_c = l_flat[sl]   # (C,)
                r_c = r_flat[sl]   # (C,)
 
                if name == "binary":
                    # Accumulate anchor pairs via np.add.at (no Python loop)
                    np.add.at(mat, (l_c, r_c), 1.0)
                    np.add.at(mat, (r_c, l_c), 1.0)
 
                elif name == "loop_fill":
                    # Build indicator vectors in one broadcast, then outer-sum
                    # w[c, b] = 1 if l_c[c] <= b <= r_c[c] else 0  →  (C, L)
                    w = ((b[None, :] >= l_c[:, None]) &
                         (b[None, :] <= r_c[:, None])).astype(np.float32)
                    mat += w.T @ w   # (L, L)
 
                elif name == "gaussian":
                    loop_len = np.maximum(r_c - l_c, 1).astype(np.float32)  # (C,)
                    sigma    = loop_len / 3.0                                  # (C,)
                    center   = (l_c + r_c) / 2.0                              # (C,)
                    # (C, L) weight matrix
                    w = np.exp(-0.5 * ((b[None, :] - center[:, None]) /
                                       sigma[:, None]) ** 2).astype(np.float32)
                    # Zero out beads outside [l_c, r_c]
                    w *= ((b[None, :] >= l_c[:, None]) &
                          (b[None, :] <= r_c[:, None]))
                    mat += w.T @ w   # (L, L)  — single BLAS call
 
                elif name == "tanh":
                    loop_len  = np.maximum(r_c - l_c, 1).astype(np.float32)
                    sharpness = np.maximum(loop_len * 0.05, 1.0)              # (C,)
                    # (C, L)
                    w = (np.tanh((b[None, :] - l_c[:, None] + 0.5) /
                                 sharpness[:, None]) *
                         np.tanh((r_c[:, None] - b[None, :] + 0.5) /
                                 sharpness[:, None])).astype(np.float32)
                    w = np.clip(w, 0.0, None)
                    mat += w.T @ w
 
                pbar.update(l_c.size)
 
        results[name] = mat / T
 
    if viz:
        _plot_and_save(results, path)
 
    return results
 
 
def _plot_and_save(results, path):
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)
 
    # Red colormap: white → pink → red → dark red (classic Hi-C look)
    hic_red = mcolors.LinearSegmentedColormap.from_list(
        "hic_red", ["#ffffff", "#ffcccc", "#ff4444", "#cc0000", "#6b0000"]
    )
 
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.2), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")
 
    for ax, (name, mat) in zip(axes, results.items()):
        title, _ = _META[name]   # cmap from _META ignored; always use hic_red
 
        eps  = mat[mat > 0].min() * 0.01 if mat.any() else 1e-6
        norm = mcolors.LogNorm(vmin=eps, vmax=mat.max() + eps)
 
        im = ax.imshow(mat, cmap=hic_red, norm=norm, origin="upper",
                       aspect="equal", interpolation="nearest")
 
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="black")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="black", fontsize=7)
        cb.set_label("Contact frequency", color="black", fontsize=8)
        cb.outline.set_edgecolor("black")
 
        ax.set_title(title, color="black", fontsize=10, pad=8)
        ax.set_xlabel("Bead index", color="black", fontsize=8)
        ax.set_ylabel("Bead index", color="black", fontsize=8)
        ax.tick_params(colors="black", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
        ax.set_facecolor("white")
 
    fig.suptitle(
        "Stochastic Hi-C — LEF trajectory-averaged contact maps\n"
        f"[{', '.join(results)}]",
        color="black", fontsize=12, y=1.02, fontweight="bold",
    )
    plt.tight_layout()
 
    plots_dir = os.path.join(path, "plots")
    for fmt, dpi in [("svg", 600), ("pdf", 600), ("png", 600)]:
        out = os.path.join(plots_dir, f"stochastic_heatmap.{fmt}")
        plt.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out}")
 
    plt.close(fig)



def combine_matrices(path_upper,path_lower,label_upper,label_lower,th1=0,th2=50,color="Reds"):
    mat1 = np.load(path_upper)
    mat2 = np.load(path_lower)
    mat1 = mat1/np.average(mat1)*10
    mat2 = mat2/np.average(mat2)*10
    L1 = len(mat1)
    L2 = len(mat2)

    ratio = 1
    if L1!=L2:
        if L1>L2:
            mat1 = average_pooling(mat1,dim_new=L2)
            ratio = L1//L2
        else:
            mat2 = average_pooling(mat2,dim_new=L1)
            
    print('1 pixel of heatmap corresponds to {} bp'.format(ratio*5000))
    exp_tr = np.triu(mat1)
    sim_tr = np.tril(mat2)
    full_m = exp_tr+sim_tr

    arialfont = {'fontname':'Arial'}

    figure(figsize=(10, 10))
    plt.imshow(full_m ,cmap=color,vmin=th1,vmax=th2)
    plt.text(750,250,label_upper,ha='right',va='top',fontsize=30)
    plt.text(250,750,label_lower,ha='left',va='bottom',fontsize=30)
    # plt.xlabel('Genomic Distance (x5kb)',fontsize=16)
    # plt.ylabel('Genomic Distance (x5kb)',fontsize=16)
    plt.xlabel('Genomic Distance (x5kb)',fontsize=20)
    plt.ylabel('Genomic Distance (x5kb)',fontsize=20)
    plt.savefig('comparison_reg3.png',format='png',dpi=200)
    plt.close()
