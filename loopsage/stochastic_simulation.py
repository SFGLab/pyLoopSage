#Basic Libraries
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.stats as stats
from numba import njit, prange
from tqdm import tqdm
import importlib.resources

# scipy
from scipy.stats import norm
from scipy.stats import poisson

# My own libraries
from .preproc import *
from .plots import *
from .md import *
from .em import *

# Dynamically set the default path to the XML file in the package
try:
    with importlib.resources.path('loopsage.forcefields', 'classic_sm_ff.xml') as default_xml_path:
        default_xml_path = str(default_xml_path)
except FileNotFoundError:
    # If running in a development setup without the resource installed, fallback to a relative path
    default_xml_path = 'loopsage/forcefields/classic_sm_ff.xml'

@njit
def Kappa(mi,ni,mj,nj,cross_loop=True):
    '''
    Computes the crossing function of LoopSage.
    '''
    k=0.0
    if cross_loop:
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
def E_cross(ms, ns, k_norm, N_lef, cross_loop=True, between_families_penalty=True):
    '''
    The crossing energy.
    '''
    crossing = 0.0
    for i in prange(len(ms)):
        for j in range(i + 1, len(ms)):
            if between_families_penalty or (i < N_lef and j < N_lef) or (i >= N_lef and j >= N_lef):
                crossing += Kappa(ms[i], ns[i], ms[j], ns[j], cross_loop)
    return k_norm * crossing

@njit
def E_fold(ms, ns, fold_norm):
    ''''
    The folding energy.
    '''
    folding = np.sum(np.log(ns - ms))
    return fold_norm * folding

@njit
def E_bw(N_bws, r, BWs, ms, ns):
    '''
    Calculation of the RNApII binding energy. Needs cohesins positions as input.
    '''
    E_bw = 0
    for i in range(N_bws):
        E_bw += r[i] * np.sum(BWs[i, ms] + BWs[i, ns]) / np.sum(BWs[i])
    return E_bw

@njit
def get_E(L, R, bind_norm, fold_norm, fold_norm2, k_norm, ms, ns, N_lef, N_lef2, cross_loop, r=None, N_bws=0, BWs=None, between_families_penalty=True):
    ''''
    The total energy.
    '''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm, cross_loop, between_families_penalty) + E_fold(ms, ns, fold_norm)
    if fold_norm2!=0: energy += E_fold(ms[N_lef:N_lef+N_lef2],ns[N_lef:N_lef+N_lef2], fold_norm2)
    if r is not None and BWs is not None: energy += E_bw(N_bws, r, BWs, ms, ns)
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
def get_dE_bw(N_bws, r, BWs, ms, ns, m_new, n_new, idx):
    dE_bw = 0
    for i in range(N_bws):
        dE_bw += r[i] * (BWs[i, m_new] + BWs[i, n_new] - BWs[i, ms[idx]] - BWs[i, ns[idx]]) / np.sum(BWs[i])
    return dE_bw

@njit
def get_dE_cross(ms, ns, m_new, n_new, idx, k_norm, cross_loop, N_lef, between_families_penalty):
    '''
    Energy difference for crossing energy.
    '''
    K1, K2 = 0, 0
    for i in prange(len(ms)):
        if i != idx:
            if between_families_penalty or (idx < N_lef and i < N_lef) or (idx >= N_lef and i >= N_lef):
                K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i], cross_loop)
                K2 += Kappa(m_new, n_new, ms[i], ns[i], cross_loop)
    return k_norm * (K2 - K1)

@njit
def get_dE(L, R, bind_norm, fold_norm, fold_norm2, k_norm, ms, ns, m_new, n_new, idx, N_lef, N_lef2, cross_loop, r=None, N_bws=0, BWs=None, between_families_penalty=True):
    '''
    Total energy difference.
    '''
    dE = 0.0
    if idx<N_lef:
        dE += get_dE_fold(fold_norm,ms[:N_lef],ns[:N_lef],m_new,n_new,idx)
    else:
        dE += get_dE_fold(fold_norm2,ms[N_lef:N_lef+N_lef2],ns[N_lef:N_lef+N_lef2],m_new,n_new,idx-N_lef)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, k_norm, cross_loop, N_lef, between_families_penalty)
    if r is not None and BWs is not None: dE += get_dE_bw(N_bws, r, BWs, ms, ns, m_new, n_new, idx)
    return dE

@njit
def unbind_bind(N_beads, track=None):
    '''
    Rebinding Monte-Carlo step.
    '''
    if track is not None:
        weights = track / np.sum(track)
        m_new = np.searchsorted(np.cumsum(weights), np.random.rand())
    else:
        m_new = np.random.randint(0, N_beads - 3)
    n_new = m_new + 2
    return int(m_new), int(n_new)

@njit
def slide(m_old, n_old, ms, ns, N_beads, rw=True, drift=True):
    '''
    Sliding Monte-Carlo step.
    '''
    choices = np.array([-1, 1], dtype=np.int64)
    r1 = np.random.choice(choices) if rw else -1
    r2 = np.random.choice(choices) if rw else 1
    m_new = max(m_old + r1, 0)
    if np.any(ns == m_new) and drift and m_old - r1 < n_old - 1: 
        m_new = max(m_old - r1, 0)
    n_new = min(n_old + r2, N_beads - 1)
    if np.any(ms == n_new) and drift and n_old - r2 > m_old + 1: 
        n_new = min(n_old - r2, N_beads - 1)
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
def initialize(N_beads, N_lef, track=None):
    '''
    Random initialization of polymer DNA fiber with some cohesin positions.
    '''
    ms, ns = np.zeros(N_lef,dtype=np.int64), np.zeros(N_lef,dtype=np.int64)
    for i in range(N_lef):
        ms[i], ns[i] = unbind_bind(N_beads, track)
    return ms, ns

@njit
def run_simulation(N_beads, N_steps, MC_step, burnin, T, T_min, fold_norm, fold_norm2, bind_norm, k_norm, N_lef, N_lef2, L, R, mode, lef_rw=True, lef_drift=True, cross_loop=True, r=None, N_bws=0, BWs=None, track=None, between_families_penalty=True):
    '''
    Runs the Monte Carlo simulation.
    '''
    Ti = T
    bi = burnin // MC_step
    ms, ns = initialize(N_beads, N_lef + N_lef2, track)
    E = get_E(L, R, bind_norm, fold_norm, fold_norm2, k_norm, ms, ns, N_lef, N_lef2, cross_loop, r, N_bws, BWs, between_families_penalty)
    Es, Ks, Fs, Bs, ufs = np.zeros(N_steps // MC_step, dtype=np.float64), np.zeros(N_steps // MC_step, dtype=np.float64), np.zeros(N_steps // MC_step, dtype=np.float64), np.zeros(N_steps // MC_step, dtype=np.float64), np.zeros(N_steps // MC_step, dtype=np.float64)
    Ms, Ns = np.zeros((N_lef + N_lef2, N_steps // MC_step), dtype=np.int64), np.zeros((N_lef + N_lef2, N_steps // MC_step), dtype=np.int64)

    last_percent = -1

    for i in range(N_steps):
        # Print progress every 5%
        percent = int(100 * i / N_steps)
        if percent % 5 == 0 and percent != last_percent:
            # Numba can't use print with flush, so just print
            print(f"Progress: {percent} % completed.")
            last_percent = percent
        
        Ti = T - (T - T_min) * (i + 1) / N_steps if mode == 'Annealing' else T
        for j in range(N_lef + N_lef2):
            # Randomly choose a move (sliding or rebinding)
            r_move = np.random.choice(np.array([0, 1]))
            if r_move == 0:
                m_new, n_new = unbind_bind(N_beads, track)
            else:
                m_new, n_new = slide(ms[j], ns[j], ms, ns, N_beads, lef_rw, lef_drift)

            # Compute energy difference
            dE = get_dE(L, R, bind_norm, fold_norm, fold_norm2, k_norm, ms, ns, m_new, n_new, j, N_lef, N_lef2, cross_loop, r, N_bws, BWs, between_families_penalty)
            
            if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                ms[j], ns[j] = m_new, n_new
                E += dE
            # Compute Metrics
            if i % MC_step == 0:
                Ms[j, i // MC_step], Ns[j, i // MC_step] = ms[j], ns[j]
            
        # Compute Metrics
        if i % MC_step == 0:
            ufs[i // MC_step] = unfolding_metric(ms, ns, N_beads)
            Es[i // MC_step] = E
            Ks[i // MC_step] = E_cross(ms, ns, k_norm, cross_loop, N_lef, between_families_penalty)
            Fs[i // MC_step] = E_fold(ms, ns, fold_norm)
            Bs[i // MC_step] = E_bind(L, R, ms, ns, bind_norm)
    return Ms, Ns, Es, Ks, Fs, Bs, ufs

class StochasticSimulation:
    def __init__(self,region,chrom,bedpe_file,N_beads=None,N_lef=None,N_lef2=0,out_dir=None, bw_files=None, track_file=None):
        '''
        Definition of simulation parameters and input files.
        
        region (list): [start,end].
        chrom (str): indicator of chromosome.
        bedpe_file (str): path where is the bedpe file with CTCF loops.
        N_beads (int): number of monomers in the polymer chain.
        N_lef (int): number of cohesins in the system.
        kappa (float): LEF crossing coefficient of Hamiltonian.
        f (float): folding coeffient of Hamiltonian.
        b (float): binding coefficient of Hamiltonian.
        r (list): strength of each ChIP-Seq experinment.
        '''
        self.N_beads = N_beads if N_beads!=None else int(np.round((region[1]-region[0])/2000))
        self.N_bws = len(bw_files) if bw_files else 0
        print('Number of beads:',self.N_beads)
        self.chrom, self.region = chrom, region
        self.bedpe_file, self.bw_files, self.track_file = bedpe_file, bw_files, track_file
        self.preprocessing()
        self.N_lef = 2*self.N_CTCF if N_lef==None else N_lef
        self.N_lef2 = N_lef2
        print('Number of LEFs:',self.N_lef+self.N_lef2)
        self.path = make_folder(out_dir)
    
    def run_energy_minimization(self, N_steps, MC_step, burnin, T=1, T_min=0, mode='Metropolis', viz=False, save=False, f=1.0, f2=0.0, b=1.0, kappa=1.0, lef_rw=True, lef_drift=True, cross_loop=True, r=None, between_families_penalty=True):
        '''
        Implementation of the stochastic Monte Carlo simulation.

        Input parameters:
        N_steps (int): number of Monte Carlo steps.
        MC_step (int): sampling frequency.
        burnin (int): definition of the burnin period.
        T (float): simulation (initial) temperature.
        mode (str): it can be either 'Metropolis' or 'Annealing'.
        viz (bool): True in case that user wants to see plots.
        r (list): strength of each ChIP-Seq experiment.
        N_bws (int): number of binding weight matrices.
        BWs (np.ndarray): binding weight matrices.
        between_families_penalty (bool): whether to apply penalty for interactions between families.
        '''
        # Define normalization constants
        fold_norm, fold_norm2, bind_norm, k_norm = -self.N_beads*f/((self.N_lef+self.N_lef2)*np.log(self.N_beads/(self.N_lef+self.N_lef2))), -self.N_beads*f2/((self.N_lef+self.N_lef2)*np.log(self.N_beads/(self.N_lef+self.N_lef2))), -self.N_beads*b/(np.sum(self.L)+np.sum(self.R)), kappa*1e4
        self.N_steps, self.MC_step = N_steps, MC_step
        r = np.full(self.N_bws, -self.N_beads / 10) if not r and self.N_bws > 0 else (None if not r else r)

        # Run simulation
        print('\nRunning simulation (with numba acceleration)...')
        start = time.time()
        self.burnin = burnin
        self.Ms, self.Ns, self.Es, self.Ks, self.Fs, self.Bs, self.ufs = run_simulation(self.N_beads, N_steps, MC_step, burnin, T, T_min, fold_norm, fold_norm2, bind_norm, k_norm, self.N_lef, self.N_lef2, self.L, self.R, mode, lef_rw, lef_drift, cross_loop, r, self.N_bws, self.BWs, self.lef_track, between_families_penalty)
        end = time.time()
        elapsed = end - start
        print(f'Computation finished successfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and {elapsed%60:.0f} seconds.')
        
        # Save simulation info
        if save:
            save_dir = os.path.join(self.path, 'other') + '/'
            with open(save_dir + 'info.txt', "w") as file:
                file.write(f'Number of beads {self.N_beads}.\n')
                file.write(f'Number of cohesins {self.N_lef}. Number of cohesins in second family {self.N_lef2}. Number of CTCFs {self.N_CTCF}. \n')
                file.write(f'Bedpe file for CTCF binding is {self.bedpe_file}.\n')
                file.write(f'Initial temperature {T}. Minimum temperature {T_min}.\n')
                file.write(f'Monte Carlo optimization method: {mode}.\n')
                file.write(f'Monte Carlo steps {N_steps}. Sampling frequency {self.MC_step}. Burnin period {burnin}.\n')
                file.write(f'Crossing energy in equilibrium is {np.average(self.Ks[burnin//MC_step:]):.2f}. Crossing coefficient kappa={kappa}.\n')
                file.write(f'Folding energy in equilibrium is {np.average(self.Fs[burnin//MC_step:]):.2f}. Folding coefficient f={f}. Folding coefficient for the second family f2={f2}\n')
                file.write(f'Binding energy in equilibrium is {np.average(self.Bs[burnin//MC_step:]):.2f}. Binding coefficient b={b}.\n')
                if r is not None and self.BWs is not None:
                    file.write(f'RNApII binding energy included with {self.N_bws} binding weight matrices.\n')
                file.write(f'Energy at equilibrium: {np.average(self.Es[self.burnin//MC_step:]):.2f}.\n')
            np.save(save_dir + 'Ms.npy', self.Ms)
            np.save(save_dir + 'Ns.npy', self.Ns)
            np.save(save_dir + 'ufs.npy', self.ufs)
            np.save(save_dir + 'Es.npy', self.Es)
            np.save(save_dir + 'Bs.npy', self.Bs)
            np.save(save_dir + 'Fs.npy', self.Fs)
            np.save(save_dir + 'Ks.npy', self.Ks)
        
        # Some visualizations
        if viz: coh_traj_plot(self.Ms, self.Ns, self.N_beads, self.path)
        if viz: make_timeplots(self.Es, self.Bs, self.Ks, self.Fs, burnin//MC_step, mode, self.path)
        if viz: coh_probdist_plot(self.Ms, self.Ns, self.N_beads, self.path)
        if viz and self.N_beads <= 2000: stochastic_heatmap(self.Ms, self.Ns, MC_step, self.N_beads, self.path)
        
        return self.Es, self.Ms, self.Ns, self.Bs, self.Ks, self.Fs, self.ufs

    def preprocessing(self):
        self.L, self.R, self.dists = binding_vectors_from_bedpe(self.bedpe_file,self.N_beads,self.region,self.chrom,False,False)
        if not self.bw_files:
            self.BWs = None
            self.N_bws = 0
        else:
            if isinstance(self.bw_files, str):
                self.bw_files = [self.bw_files]
                self.N_bws = 1
            self.BWs = np.zeros((self.N_bws, self.N_beads))
            for i, f in enumerate(self.bw_files):
                self.BWs[i, :] = load_track(file=f, region=self.region, chrom=self.chrom, N_beads=self.N_beads, viz=False)
        self.lef_track = load_track(self.track_file,self.region,self.chrom,self.N_beads,False,True) if self.track_file else None
        self.N_CTCF = np.max([np.count_nonzero(self.L),np.count_nonzero(self.R)])
        print('Number of CTCF:',self.N_CTCF)

    def run_EM(self,platform='CPU',angle_ff_strength=200,le_distance=0.1,le_ff_strength=50000.0,ev_ff_strength=100.0,ev_ff_power=3.0,tolerance=0.001,friction=0.1,integrator_step=100*mm.unit.femtosecond,temperature=310,init_struct='rw',save_plots=True,ff_path=default_xml_path):
        em = EM_LE(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.path,platform,angle_ff_strength,le_distance,le_ff_strength,ev_ff_strength,ev_ff_power,tolerance)
        sim_heat = em.run_pipeline(plots=save_plots,friction=friction,integrator_step=integrator_step,temperature=temperature,ff_path=ff_path,init_struct=init_struct)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)
    
    def run_MD(self,platform='CPU',angle_ff_strength=200,le_distance=0.1,le_ff_strength=50000.0,ev_ff_strength=100.0,ev_ff_power=3.0,tolerance=0.001,friction=0.1,integrator_step=100*mm.unit.femtosecond,temperature=310,init_struct='rw',sim_step=1000,save_plots=True,ff_path=default_xml_path,p_ev=0,continuous_topoisomerase=False):
        md = MD_LE(self.Ms,self.Ns,self.N_beads,self.path,platform,angle_ff_strength,le_distance,le_ff_strength,ev_ff_strength,ev_ff_power,tolerance)
        sim_heat = md.run_pipeline(plots=save_plots,sim_step=sim_step,friction=friction,integrator_step=integrator_step,temperature=temperature,ff_path=ff_path,p_ev=p_ev,init_struct=init_struct,continuous_topoisomerase=continuous_topoisomerase)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)

def main():
    # Definition of Monte Carlo parameters
    N_steps, MC_step, burnin, T, T_min = int(4e4), int(5e2), 1000, 3.0, 1.0
    N_lef, N_lef2 = 100, 20
    lew_rw = True
    mode = 'Annealing'
    
    # Simulation Strengths
    f, f2, b, kappa = 1.0, 2.0, 1.0, 1.0
    
    # Definition of region
    region, chrom = [15550000, 16850000], 'chr6'
    
    # Definition of data
    output_name = 'tmp'
    bedpe_file = '/home/skorsak/Data/HiChIP/Maps/hg00731_smc1_maps_2.bedpe'
    
    # Between family penalty
    between_families_penalty = True
    
    sim = StochasticSimulation(region, chrom, bedpe_file, out_dir=output_name, N_beads=1000, N_lef=N_lef, N_lef2=N_lef2)
    Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(
        N_steps, MC_step, burnin, T, T_min, mode=mode, viz=True, save=True, f=f, f2=f2, b=b, kappa=kappa, lef_rw=lew_rw, between_families_penalty=between_families_penalty
    )
    sim.run_MD('CUDA', continuous_topoisomerase=True, p_ev=0.01)

if __name__=='__main__':
    main()