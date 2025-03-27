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

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    elif os.path.isdir(folder_name):
        print(f'Directory with name "{folder_name}" already exists! No problem lets continue!')
    else:
        raise IOError(f'File with name "{folder_name}" already exists! Please change the name of the folder!')        
    for subfolder in ['plots', 'other', 'ensemble']:
        os.makedirs(os.path.join(folder_name, subfolder), exist_ok=True)
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

def corr_exp_heat(mat_sim,bedpe_file,region,chrom,N_beads,path):
    # Read file and select the region of interest
    df = pd.read_csv(bedpe_file,sep='\t',header=None)
    df = df[(df[1]>=region[0])&(df[2]>=region[0])&(df[4]>=region[0])&(df[5]>=region[0])&(df[5]<region[1])&(df[4]<region[1])&(df[1]<region[1])&(df[2]<region[1])&(df[0]==chrom)].reset_index(drop=True)

    # Convert hic coords into simulation beads
    resolution = (region[1]-region[0])//N_beads
    df[1], df[2], df[4], df[5] = (df[1]-region[0])//resolution, (df[2]-region[0])//resolution, (df[4]-region[0])//resolution, (df[5]-region[0])//resolution
    
    # Compute the matrix
    exp_vec, th_vec = np.zeros(N_beads), np.zeros(N_beads)
    for i in range(len(df)):
        x, y = (df[1][i]+df[2][i])//2, (df[4][i]+df[5][i])//2
        exp_vec[x]+=df[6][i]
        exp_vec[y]+=df[6][i]
        th_vec[x]+=mat_sim[x,y]
        th_vec[y]+=mat_sim[x,y]        

    # pearson correlation calculation
    pears, pval1 = pearsonr(th_vec,exp_vec)
    spear, pval2 = spearmanr(th_vec,exp_vec)
    kendal, pval3 = kendalltau(th_vec,exp_vec)
    print('-------------- Optimistic Correlation -----------')
    print(f'Pearson Correlation with loops strengths: {pears:.3f} with pvalue {pval1}.')
    print(f'Spearman Correlation with loops strengths: {spear:.3f} with pvalue {pval2}.')
    print(f'Kendall Correlation with loops strengths: {kendal:.3f} with pvalue {pval3}.\n')

    f = open(path+'/other/correlations.txt', "w")
    f.write('---- Optimistic Estimations ----\n')
    f.write(f'Pearson Correlation with experimental heatmap: {pears:.3f} with pvalue {pval1}.\n')
    f.write(f'Spearman Correlation with experimental heatmap: {spear:.3f} with pvalue {pval2}.\n')
    f.write(f'Kendall Correlation with experimental heatmap: {kendal:.3f} with pvalue {pval3}.\n\n')    

    mask1, mask2 = exp_vec==0, th_vec==0
    exp_vec, th_vec = exp_vec[~mask1], th_vec[~mask2]
    pears, pval1 = pearsonr(th_vec,exp_vec)
    spear, pval2 = spearmanr(th_vec,exp_vec)
    kendal, pval3 = kendalltau(th_vec,exp_vec)

    print('-------------- Pessimistic Correlation -----------')
    print(f'Pearson Correlation with loops strengths: {pears:.3f} with pvalue {pval1}.')
    print(f'Spearman Correlation with loops strengths: {spear:.3f} with pvalue {pval2}.')
    print(f'Kendall Correlation with loops strengths: {kendal:.3f} with pvalue {pval3}.\n')

    f.write('---- Pessimistic Estimations ----\n')
    f.write(f'Pearson Correlation with experimental heatmap: {pears:.3f} with pvalue {pval1}.\n')
    f.write(f'Spearman Correlation with experimental heatmap: {spear:.3f} with pvalue {pval2}.\n')
    f.write(f'Kendall Correlation with experimental heatmap: {kendal:.3f} with pvalue {pval3}.\n\n')
    f.close()

    fig, axs = plt.subplots(2, figsize=(15, 10))
    fig.suptitle(f'Estimated Pearson Correlation {pears:.3f}',fontsize=18)
    axs[0].plot(exp_vec)
    axs[0].set_ylabel('Experimental Signal',fontsize=16)
    axs[1].plot(th_vec)
    axs[1].set_ylabel('Simulation Signal',fontsize=16)
    axs[1].set_xlabel('Genomic Distance (with simumation beads as a unit)',fontsize=16)
    fig.savefig(path+'/plots/pearson.png',dpi=600)
    fig.savefig(path+'/plots/pearson.pdf',dpi=600)
    plt.close()

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
