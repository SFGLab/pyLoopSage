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

def correlation_plot(given_heatmap,T_range,path):
    pearsons, spearmans, kendals = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
    exp_heat_dim = len(given_heatmap)
    for i, T in enumerate(T_range):
        N_beads,N_coh,kappa,f,b = 500,30,20000,-2000,-2000
        N_steps, MC_step, burnin = int(1e4), int(1e2), 20
        L, R = binding_vectors_from_bedpe_with_peaks("/mnt/raid/data/Zofia_Trios/bedpe/hg00731_CTCF_pulled_2.bedpe",N_beads,[178421513,179491193],'chr1',False)
        sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R)
        Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,mode='Metropolis',viz=True,vid=False)
        md = MD_LE(Ms,Ns,N_beads,burnin,MC_step)
        heat = md.run_pipeline(write_files=False,plots=False)
        if N_beads>exp_heat_dim:
            heat = average_pooling(heat,exp_heat_dim)
            L = exp_heat_dim
        else:
            given_heatmap = average_pooling(given_heatmap,N_beads)
            L = N_beads
        a, b = np.reshape(heat, (L**2, )), np.reshape(given_heatmap, (L**2, ))
        pearsons[i] = scipy.stats.pearsonr(a,b)[0]
        spearmans[i] = scipy.stats.spearmanr(a, b).correlation
        kendals[i] = scipy.stats.kendalltau(a, b).correlation
        print(f'\nTemperature:{T}, Pearson Correlation coefficient:{pearsons[i]}, Spearman:{spearmans[i]}, Kendal:{kendals[i]}\n\n')

    figure(figsize=(10, 8), dpi=600)
    plt.plot(T_range,pearsons,'bo-')
    plt.plot(T_range,spearmans,'ro-')
    plt.plot(T_range,kendals,'go-')
    # plt.plot(T_range,Cross,'go-')
    plt.ylabel('Correlation with Experimental Heatmap', fontsize=16)
    plt.xlabel('Temperature', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(['Pearson','Spearman','Kendall Tau'])
    plt.grid()
    save_path = path+'/plots/pearson_plot.pdf' if path!=None else 'pearson_plot.pdf'
    plt.savefig(save_path,dpi=600)
    plt.close()

def coh_traj_plot(ms,ns,N_beads,path):
    N_coh = len(ms)
    figure(figsize=(18, 25))
    color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) for i in range(N_coh)]
    size = 0.01 if (N_beads > 500 or N_coh > 20) else 0.1
    
    ls = 'None'
    for nn in range(N_coh):
        tr_m, tr_n = ms[nn], ns[nn]
        plt.fill_between(np.arange(len(tr_m)), tr_m, tr_n, color=color[nn], alpha=0.4, interpolate=False, linewidth=0)
    plt.xlabel('Simulation Step', fontsize=24)
    plt.ylabel('Position of Cohesin', fontsize=24)
    plt.gca().invert_yaxis()
    save_path = path+'/plots/coh_trajectories.png' if path!=None else 'coh_trajectories.png'
    plt.savefig(save_path, format='png', dpi=200)
    save_path = path+'/plots/coh_trajectories.svg' if path!=None else 'coh_trajectories.svg'
    plt.savefig(save_path, format='svg', dpi=200)
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
