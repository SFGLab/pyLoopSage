#Basic Libraries
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.stats as stats
from numba import njit, prange
from tqdm import tqdm

# scipy
from scipy.stats import norm
from scipy.stats import poisson

# My own libraries
from .preproc import *
from .plots import *
from .md import *
from .em import *

@njit
def Kappa(mi,ni,mj,nj):
    '''
    Computes the crossing function of LoopSage.
    '''
    k=0.0
    if mi<mj and mj<ni and ni<nj: k+=1 # np.abs(ni-mj)+1
    if mj<mi and mi<nj and nj<ni: k+=1 # np.abs(nj-mi)+1
    if mj==ni or mi==nj or ni==nj or mi==mj: k+=1
    return k

@njit
def E_bind(L, R, ms, ns, bind_norm):
    '''
    The binding energy.
    '''
    binding = np.sum(L[ms] + R[ns])
    E_b = bind_norm * binding
    return E_b

@njit
def E_cross(ms, ns, k_norm):
    '''
    The crossing energy.
    '''
    crossing = 0.0
    for i in prange(len(ms)):
        for j in range(i + 1, len(ms)):
            crossing += Kappa(ms[i], ns[i], ms[j], ns[j])
    return k_norm * crossing

@njit
def E_fold(ms, ns, fold_norm):
    ''''
    The folding energy.
    '''
    folding = np.sum(np.log(ns - ms))
    return fold_norm * folding

@njit
def get_E(L, R, bind_norm, fold_norm, k_norm, ms, ns):
    ''''
    The total energy.
    '''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm) + E_fold(ms, ns, fold_norm)
    return energy

@njit
def get_dE_bind(L,R,bind_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for binding energy.
    '''
    return bind_norm*(L[m_new]+R[n_new]-L[ms[idx]]-R[ns[idx]])
    
@njit
def get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for folding energy.
    '''
    return fold_norm*(np.log(n_new-m_new)-np.log(ns[idx]-ms[idx]))

@njit
def get_dE_cross(ms, ns, m_new, n_new, idx, k_norm):
    '''
    Energy difference for crossing energy.
    '''
    K1, K2 = 0, 0
    for i in prange(len(ms)):
        if i != idx:
            K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i])
            K2 += Kappa(m_new, n_new, ms[i], ns[i])
    return k_norm * (K2 - K1)

@njit
def get_dE(L, R, bind_norm, fold_norm, k_norm, ms, ns, m_new, n_new, idx):
    '''
    Total energy difference.
    '''
    dE = 0.0
    dE += get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, k_norm)
    return dE

@njit
def unbind_bind(N_beads):
    '''
    Rebinding Monte-Carlo step.
    '''
    m_new = rd.randint(0, N_beads - 3)
    n_new = m_new + 2
    return int(m_new), int(n_new)

@njit
def slide(m_old, n_old, N_beads, rw=True):
    '''
    Sliding Monte-Carlo step.
    '''
    choices = np.array([-1, 1], dtype=np.int64)
    r1 = np.random.choice(choices) if rw else -1
    r2 = np.random.choice(choices) if rw else 1
    m_new = m_old + r1 if m_old + r1>=0 else 0
    n_new = n_old + r2 if n_old + r2<N_beads else N_beads-1
    return int(m_new), int(n_new)

@njit
def unfolding_metric(ms,ns,N_beads):
    '''
    This is a metric for the number of gaps (regions unfolded that are not within a loop).
    Cohesin positions are needed as input.
    '''
    fiber = np.zeros(N_beads)
    for i in range(len(ms)):
        fiber[ms[i]:ns[i]]=1
    unfold = 2*(N_beads-np.count_nonzero(fiber))/N_beads
    return unfold

@njit
def initialize(N_beads,N_lef):
    '''
    Random initialization of polymer DNA fiber with some cohesin positions.
    '''
    ms, ns = np.zeros(N_lef,dtype=np.int64), np.zeros(N_lef,dtype=np.int64)
    for i in range(N_lef):
        ms[i], ns[i] = unbind_bind(N_beads)
    return ms, ns

@njit
def run_simulation(N_beads,N_steps,MC_step,burnin,T,T_min,fold_norm,bind_norm,k_norm,N_lef,L,R,mode,lef_rw=True):
    '''
    Runs the Monte Carlo simulation.
    '''
    Ti = T
    bi = burnin//MC_step
    ms, ns = initialize(N_beads,N_lef)
    E = get_E(L, R, bind_norm, fold_norm, k_norm,  ms, ns)
    Es,Ks,Fs,Bs,ufs = np.zeros(N_steps//MC_step, dtype=np.float64),np.zeros(N_steps//MC_step, dtype=np.float64),np.zeros(N_steps//MC_step, dtype=np.float64),np.zeros(N_steps//MC_step, dtype=np.float64),np.zeros(N_steps//MC_step, dtype=np.float64)
    Ms, Ns = np.zeros((N_lef,N_steps//MC_step), dtype=np.int64), np.zeros((N_lef,N_steps//MC_step), dtype=np.int64)

    for i in range(N_steps):
        Ti = T-(T-T_min)*(i+1)/N_steps if mode=='Annealing' else T
        for j in range(N_lef):
            # Randomly choose a move (sliding or rebinding)
            r = np.random.choice(np.array([0, 1]))
            if r==0:
                m_new, n_new = unbind_bind(N_beads)
            else:
                m_new, n_new = slide(ms[j],ns[j],N_beads,lef_rw)

            # Compute energy difference
            dE = get_dE(L, R, bind_norm, fold_norm, k_norm, ms, ns, m_new, n_new,j)
            if dE <= 0 or np.exp(-dE/Ti) > np.random.rand():
                ms[j], ns[j] = m_new, n_new
                E += dE
            # Compute Metrics
            if i%MC_step==0:
                Ms[j,i//MC_step], Ns[j,i//MC_step] = ms[j], ns[j]
            
        # Compute Metrics
        if i%MC_step==0:
            ufs[i//MC_step] = unfolding_metric(ms,ns,N_beads)
            Es[i//MC_step] = E
            Ks[i//MC_step] = E_cross(ms,ns,k_norm)
            Fs[i//MC_step] = E_fold(ms,ns,fold_norm)
            Bs[i//MC_step] = E_bind(L,R,ms,ns,bind_norm)
    return Ms, Ns, Es, Ks, Fs, Bs, ufs

class StochasticSimulation:
    def __init__(self,region,chrom,bedpe_file,N_beads=None,N_lef=None,track_file=None,bw_files=None,out_dir=None):
        '''
        Definition of simulation parameters and input files.
        
        region (list): [start,end].
        chrom (str): indicator of chromosome.
        bedpe_file (str): path where is the bedpe file with CTCF loops.
        track_file (str): bigwig file with cohesin coverage.
        bw_files (list of str): bigwig files with (or other protein of interest) coverage.
        N_beads (int): number of monomers in the polymer chain.
        N_lef (int): number of cohesins in the system.
        kappa (float): LEF crossing coefficient of Hamiltonian.
        f (float): folding coeffient of Hamiltonian.
        b (float): binding coefficient of Hamiltonian.
        r (list): strength of each ChIP-Seq experinment.
        '''
        self.N_beads = N_beads if N_beads!=None else int(np.round((region[1]-region[0])/2000))
        print('Number of beads:',self.N_beads)
        self.chrom, self.region = chrom, region
        self.bedpe_file, self.track_file = bedpe_file, track_file
        self.preprocessing()
        self.N_lef = 2*self.N_CTCF if N_lef==None else N_lef
        print('Number of LEFs:',self.N_lef)
        self.path = make_folder(out_dir)
    
    def run_energy_minimization(self,N_steps,MC_step,burnin,T=1,T_min=0,mode='Metropolis',viz=False,save=False, f=1.0, b=1.0, kappa=1.0, lef_rw=True):
        '''
        Implementation of the stochastic Monte Carlo simulation.

        Input parameters:
        N_steps (int): number of Monte Carlo steps.
        MC_step (int): sampling frequency.
        burnin (int): definition of the burnin period.
        T (float): simulation (initial) temperature.
        mode (str): it can be either 'Metropolis' or 'Annealing'.
        viz (bool): True in case that user wants to see plots.
        '''
        # Define normalization constants
        fold_norm, bind_norm, k_norm = -self.N_beads*f/(self.N_lef*np.log(self.N_beads/self.N_lef)), -self.N_beads*b/(np.sum(self.L)+np.sum(self.R)), kappa*1e4
        self.N_steps, self.MC_step = N_steps, MC_step

        # Run simulation
        print('\nRunning simulation (with parallelization across CPU cores)...')
        start = time.time()
        self.burnin = burnin
        self.Ms, self.Ns, self.Es, self.Ks, self.Fs, self.Bs, self.ufs = run_simulation(self.N_beads,N_steps,MC_step,burnin,T,T_min,fold_norm,bind_norm,k_norm,self.N_lef,self.L,self.R,mode,lef_rw)
        end = time.time()
        elapsed = end - start
        print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

        # Save simulation info
        if save:
            f = open(self.path+'/other/info.txt', "w")
            f.write(f'Number of beads {self.N_beads}.\n')
            f.write(f'Number of cohesins {self.N_lef}. Number of CTCFs {self.N_CTCF}.\n')
            f.write(f'Bedpe file for CTCF binding is {self.bedpe_file}.\n')
            f.write(f'LEF track file for LEF preferential relocation is {self.track_file}.\n')
            f.write(f'Initial temperature {T}. Minimum temperature {T_min}.\n')
            f.write(f'Monte Carlo optimization method: {mode}.\n')
            f.write(f'Monte Carlo steps {N_steps}. Sampling frequency {MC_step}. Burnin period {burnin}.\n')
            f.write(f'Crossing energy in equilibrium is {np.average(self.Ks[burnin//MC_step:]):.2f}. Crossing coefficient kappa={kappa}.\n')
            f.write(f'Folding energy in equilibrium is {np.average(self.Fs[burnin//MC_step:]):.2f}. Folding coefficient f={f}.\n')
            f.write(f'Binding energy in equilibrium is {np.average(self.Bs[burnin//MC_step:]):.2f}. Binding coefficient b={b}.\n')
            f.write(f'Energy at equillibrium: {np.average(self.Es[burnin//MC_step:]):.2f}.\n')
            f.close()

            np.save(self.path+'/other/Ms.npy',self.Ms)
            np.save(self.path+'/other/Ns.npy',self.Ns)
            np.save(self.path+'/other/ufs.npy',self.ufs)
            np.save(self.path+'/other/Es.npy',self.Es)
            np.save(self.path+'/other/Bs.npy',self.Bs)
            np.save(self.path+'/other/Fs.npy',self.Fs)
            np.save(self.path+'/other/Ks.npy',self.Ks)
        
        # Some vizualizations
        if viz: coh_traj_plot(self.Ms,self.Ns,self.N_beads, self.path)
        if viz: make_timeplots(self.Es, self.Bs, self.Ks, self.Fs, burnin//MC_step, mode, self.path)
        if viz: coh_probdist_plot(self.Ms,self.Ns,self.N_beads,self.path)
        if viz and self.N_beads<=2000: stochastic_heatmap(self.Ms,self.Ns,MC_step,self.N_beads,self.path)
        
        return self.Es, self.Ms, self.Ns, self.Bs, self.Ks, self.Fs, self.ufs

    def preprocessing(self):
        self.L, self.R, self.dists = binding_vectors_from_bedpe(self.bedpe_file,self.N_beads,self.region,self.chrom,False,False)
        self.track = load_track(self.track_file,self.region,self.chrom,self.N_beads,False,True) if np.all(self.track_file!=None) else None
        self.N_CTCF = np.max([np.count_nonzero(self.L),np.count_nonzero(self.R)])
        print('Number of CTCF:',self.N_CTCF)

    def run_EM(self,platform='CPU',angle_ff_strength=200,le_distance=0.0,le_ff_strength=300000.0,ev_ff_strength=10.0,tolerance=0.001,friction=0.1,integrator_step=5*mm.unit.femtosecond,temperature=310,save_plots=True):
        em = EM_LE(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.path,platform,angle_ff_strength,le_distance,le_ff_strength,ev_ff_strength,tolerance)
        sim_heat = em.run_pipeline(plots=save_plots,friction=friction,integrator_step=integrator_step,temperature=temperature)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)
    
    def run_MD(self,platform='CPU',angle_ff_strength=200,le_distance=0.0,le_ff_strength=300000.0,ev_ff_strength=10.0,tolerance=0.001,friction=0.1,integrator_step=5*mm.unit.femtosecond,temperature=310,sim_step=100,save_plots=True):
        md = MD_LE(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.path,platform,angle_ff_strength,le_distance,le_ff_strength,ev_ff_strength,tolerance)
        sim_heat = md.run_pipeline(plots=save_plots,sim_step=sim_step,friction=friction,integrator_step=integrator_step,temperature=temperature)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)

def main():
    # Definition of Monte Carlo parameters
    N_steps, MC_step, burnin, T, T_min = int(4e4), int(5e2), 1000, 2.5, 1.0
    mode = 'Metropolis'
    
    # Simulation Strengths
    f, b, kappa = 1.0, 1.0, 1.0
    
    # Definition of region
    region, chrom = [15550000,16850000], 'chr6'
    
    # Definition of data
    output_name='../HiChIP_Annealing_T1_MD_region'
    bedpe_file = '/home/skorsak/Data/HiChIP/Maps/hg00731_smc1_maps_2.bedpe'
    
    sim = StochasticSimulation(region,chrom,bedpe_file,out_dir=output_name,N_beads=1000)
    Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,mode=mode,viz=True,save=True)
    sim.run_MD('CUDA')

if __name__=='__main__':
    main()