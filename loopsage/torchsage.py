import torch
from tqdm import tqdm
from .utils import *
from .md import *
from .em import *
from .plots import *
from .preproc import *
import numpy as np
import os
import time

class TorchSage:
    def __init__(self, N_beads, N_lef, N_lef2=0, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.N_beads = N_beads
        self.N_lef = N_lef
        self.N_lef2 = N_lef2
        self.N_total = N_lef + N_lef2
        self.ms, self.ns = self.initialize()
    
    def initialize(self):
        ms = torch.randint(0, self.N_beads - 2, (self.N_total,), device=self.device)
        ns = ms + 2
        return ms, ns
    
    def kappa(self, mi, ni, mj, nj, cross_loop=True):
        k = torch.zeros_like(mi, dtype=torch.float32)
        if cross_loop:
            k += ((mi < mj) & (mj < ni) & (ni < nj)).float()
            k += ((mj < mi) & (mi < nj) & (nj < ni)).float()
        k += ((mj == ni) | (mi == nj) | (ni == nj) | (mi == mj)).float()
        return k

    def e_bind(self, L, R, ms, ns, bind_norm):
        L = torch.tensor(L, device=self.device, dtype=torch.float32)
        R = torch.tensor(R, device=self.device, dtype=torch.float32)
        binding = torch.sum(L[ms] + R[ns])
        return bind_norm * binding

    def e_cross(self, ms, ns, k_norm, cross_loop=True):
        crossing = 0.0
        for i in range(len(ms)):
            mi, ni = ms[i], ns[i]
            mj, nj = ms[i+1:], ns[i+1:]
            crossing += torch.sum(self.kappa(mi, ni, mj, nj, cross_loop))
        return k_norm * crossing

    def e_fold(self, ms, ns, fold_norm):
        folding = torch.sum(torch.log((ns - ms).float()))
        return fold_norm * folding

    def total_energy(self, L, R, bind_norm, fold_norm, fold_norm2, k_norm, ms, ns, cross_loop=True):
        energy = self.e_bind(L, R, ms, ns, bind_norm) + self.e_cross(ms, ns, k_norm, cross_loop) + self.e_fold(ms, ns, fold_norm)
        if fold_norm2 != 0:
            energy += self.e_fold(ms[self.N_lef:], ns[self.N_lef:], fold_norm2)
        return energy

    def mc_step(self, L, R, bind_norm, fold_norm, fold_norm2, k_norm, T, cross_loop=True):
        idx = torch.randint(0, self.N_total, (1,), device=self.device).item()
        move_type = torch.randint(0, 2, (1,), device=self.device).item()
        m_old, n_old = self.ms[idx], self.ns[idx]
        
        if move_type == 0:
            m_new = torch.randint(0, self.N_beads - 2, (1,), device=self.device).item()
            n_new = m_new + 2
        else:
            shift = torch.randint(-1, 2, (1,), device=self.device).item()
            m_new = torch.clamp(m_old + shift, 0, self.N_beads - 2)
            n_new = torch.clamp(n_old + shift, 2, self.N_beads - 1)
        
        ms_new = self.ms.clone()
        ns_new = self.ns.clone()
        ms_new[idx] = m_new
        ns_new[idx] = n_new
        
        E_old = self.total_energy(L, R, bind_norm, fold_norm, fold_norm2, k_norm, self.ms, self.ns, cross_loop)
        E_new = self.total_energy(L, R, bind_norm, fold_norm, fold_norm2, k_norm, ms_new, ns_new, cross_loop)
        dE = E_new - E_old
        
        if dE <= 0 or torch.exp(-dE / T) > torch.rand(1, device=self.device):
            self.ms[idx] = m_new
            self.ns[idx] = n_new

    def run_simulation(self, L, R, N_steps, T, bind_norm, fold_norm, fold_norm2, k_norm, MC_step, cross_loop=True):
        L = torch.tensor(L, device=self.device, dtype=torch.float32)
        R = torch.tensor(R, device=self.device, dtype=torch.float32)
        energies = []
        for step in tqdm(range(N_steps)):
            self.mc_step(L, R, bind_norm, fold_norm, fold_norm2, k_norm, T, cross_loop)
            if step % MC_step == 0:
                E = self.total_energy(L, R, bind_norm, fold_norm, fold_norm2, k_norm, self.ms, self.ns, cross_loop)
                energies.append(E.item())
        return energies
    
class StochasticSimulation:
    def __init__(self,region,chrom,bedpe_file,N_beads=None,N_lef=None,N_lef2=0,out_dir=None):
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
        print('Number of beads:',self.N_beads)
        self.chrom, self.region = chrom, region
        self.bedpe_file = bedpe_file
        self.preprocessing()
        self.N_lef = 2*self.N_CTCF if N_lef==None else N_lef
        self.N_lef2 = N_lef2
        print('Number of LEFs:',self.N_lef+self.N_lef2)
        self.path = make_folder(out_dir)
    
    def run_energy_minimization(self,N_steps,MC_step,burnin,T=1,T_min=0,mode='Metropolis',viz=False,save=False, f=1.0, f2=0.0, b=1.0, kappa=1.0, cross_loop=True):
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
        fold_norm, fold_norm2, bind_norm, k_norm = -self.N_beads*f/((self.N_lef+self.N_lef2)*np.log(self.N_beads/(self.N_lef+self.N_lef2))), -self.N_beads*f2/((self.N_lef+self.N_lef2)*np.log(self.N_beads/(self.N_lef+self.N_lef2))), -self.N_beads*b/(np.sum(self.L)+np.sum(self.R)), kappa*1e4
        self.N_steps, self.MC_step = N_steps, MC_step

        # Run simulation
        print('Running simulation with torch CUDA (you are on fire bro!)...')
        start = time.time()
        self.burnin = burnin
        ts = TorchSage(self.N_beads, self.N_lef, self.N_lef2)
        ts.run_simulation(self.L, self.R, N_steps, T, bind_norm, fold_norm, fold_norm2, k_norm, MC_step, cross_loop)
        end = time.time()
        elapsed = end - start
        print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')
        
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
                file.write(f'Energy at equillibrium: {np.average(self.Es[self.burnin//MC_step:]):.2f}.\n')
            np.save(save_dir + 'Ms.npy', self.Ms)
            np.save(save_dir + 'Ns.npy', self.Ns)
            np.save(save_dir + 'ufs.npy', self.ufs)
            np.save(save_dir + 'Es.npy', self.Es)
            np.save(save_dir + 'Bs.npy', self.Bs)
            np.save(save_dir + 'Fs.npy', self.Fs)
            np.save(save_dir + 'Ks.npy', self.Ks)
        
        # Some vizualizations
        if viz: coh_traj_plot(self.Ms,self.Ns,self.N_beads, self.path)
        if viz: make_timeplots(self.Es, self.Bs, self.Ks, self.Fs, burnin//MC_step, mode, self.path)
        if viz: coh_probdist_plot(self.Ms,self.Ns,self.N_beads,self.path)
        if viz and self.N_beads<=2000: stochastic_heatmap(self.Ms,self.Ns,MC_step,self.N_beads,self.path)
        
        return self.Es, self.Ms, self.Ns, self.Bs, self.Ks, self.Fs, self.ufs

    def preprocessing(self):
        self.L, self.R, self.dists = binding_vectors_from_bedpe(self.bedpe_file,self.N_beads,self.region,self.chrom,False,False)
        self.N_CTCF = np.max([np.count_nonzero(self.L),np.count_nonzero(self.R)])
        print('Number of CTCF:',self.N_CTCF)

    def run_EM(self,platform='CPU',angle_ff_strength=200,le_distance=0.1,le_ff_strength=50000.0,ev_ff_strength=100.0,ev_ff_power=3.0,tolerance=0.001,friction=0.1,integrator_step=100*mm.unit.femtosecond,temperature=310,init_struct='rw',save_plots=True,ff_path=default_xml_path):
        em = EM_LE(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.path,platform,angle_ff_strength,le_distance,le_ff_strength,ev_ff_strength,ev_ff_power,tolerance)
        sim_heat = em.run_pipeline(plots=save_plots,friction=friction,integrator_step=integrator_step,temperature=temperature,ff_path=ff_path,init_struct=init_struct)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)
    
    def run_MD(self,platform='CPU',angle_ff_strength=200,le_distance=0.1,le_ff_strength=50000.0,ev_ff_strength=100.0,ev_ff_power=3.0,tolerance=0.001,friction=0.1,integrator_step=100*mm.unit.femtosecond,temperature=310,init_struct='rw',sim_step=1000,save_plots=True,ff_path=default_xml_path,p_ev=0):
        md = MD_LE(self.Ms,self.Ns,self.N_beads,self.path,platform,angle_ff_strength,le_distance,le_ff_strength,ev_ff_strength,ev_ff_power,tolerance)
        sim_heat = md.run_pipeline(plots=save_plots,sim_step=sim_step,friction=friction,integrator_step=integrator_step,temperature=temperature,ff_path=ff_path,p_ev=p_ev,init_struct=init_struct)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)

def main():
    # Definition of Monte Carlo parameters
    N_steps, MC_step, burnin, T, T_min = int(4e4), int(5e2), 1000, 3.0, 1.0
    N_lef, N_lef2 = 100, 20
    lew_rw=True
    mode = 'Annealing'
    
    # Simulation Strengths
    f, f2, b, kappa = 1.0, 2.0, 1.0, 1.0
    
    # Definition of region
    region, chrom = [15550000,16850000], 'chr6'
    
    # Definition of data
    output_name='../HiChIP_Annealing_T1_MD_region'
    bedpe_file = '/home/skorsak/Data/HiChIP/Maps/hg00731_smc1_maps_2.bedpe'
    
    sim = StochasticSimulation(region,chrom,bedpe_file,out_dir=output_name,N_beads=1000,N_lef=N_lef,N_lef2=N_lef2)
    sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,mode=mode,viz=True,save=True,f=f,f2=f2, b=b, kappa=kappa)
    sim.run_EM('CUDA')

if __name__=='__main__':
    main()