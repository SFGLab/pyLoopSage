import imageio
import shutil
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure
from matplotlib.pyplot import cm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats
from tqdm import tqdm
from scipy import stats

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

def coh_traj_plot(ms, ns, N_beads, path, jump_threshold=200, min_stable_time=10):
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

def stochastic_heatmap(ms,ns,step,L,path,comm_prop=True,fill_square=True):
    N_coh, N_steps = ms.shape
    mats = list()
    for t in range(0,N_steps):
        # add a loop where there is a cohesin
        mat = np.zeros((L,L))
        for m, n in zip(ms[:,t],ns[:,t]):
            mat[m,n] = 1
            mat[n,m] = 1
        
        # if a->b and b->c then a->c
        if comm_prop:
            for iter in range(3):
                xs, ys = np.nonzero(mat)
                for i, n in enumerate(ys):
                    if len(np.where(xs==(n+1))[0])>0:
                        j = np.where(xs==(n+1))[0]
                        mat[xs[i],ys[j]] = 2*iter+1
                        mat[ys[j],xs[i]] = 2*iter+1

        # feel the square that it is formed by each loop (m,n)
        if fill_square:
            xs, ys = np.nonzero(mat)
            for x, y in zip(xs,ys):
                if y>x: mat[x:y,x:y] += 0.01*mat[x,y]

        mats.append(mat)
    avg_mat = np.average(mats,axis=0)
    figure(figsize=(10, 10))
    plt.imshow(avg_mat,cmap="Reds",vmax=np.average(avg_mat)+3*np.std(avg_mat))
    save_path = path+f'/plots/stochastic_heatmap.svg' if path!=None else 'stochastic_heatmap.svg'
    plt.savefig(save_path,format='svg',dpi=200)
    # plt.colorbar()
    plt.close()

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
