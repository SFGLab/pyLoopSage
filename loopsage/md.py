#########################################################################
########### CREATOR: SEBASTIAN KORSAK, WARSAW 2022 ######################
#########################################################################

import copy
import time
import numpy as np
import openmm as mm
import openmm.unit as u
from tqdm import tqdm
from sys import stdout
from mdtraj.reporters import HDF5Reporter
from scipy import ndimage
from openmm.app import PDBFile, PDBxFile, ForceField, Simulation, PDBReporter, PDBxReporter, DCDReporter, StateDataReporter, CharmmPsfFile
from .utils import *
from .initial_structures import *

class MD_LE:
    def __init__(self,M,N,N_beads,path=None,platform='CPU',angle_ff_strength=200,le_distance=0.1,le_ff_strength=50000.0,ev_ff_strength=100.0,ev_ff_power=3.0,tolerance=0.001):
        '''
        M, N (np arrays): Position matrix of two legs of cohesin m,n. 
                          Rows represent  loops/cohesins and columns represent time
        N_beads (int): The number of beads of initial structure.
        step (int): sampling rate
        path (int): the path where the simulation will save structures etc.
        '''
        self.M, self.N = M, N
        self.N_coh, self.N_steps = M.shape
        self.N_beads = N_beads
        self.path = path if path!=None else make_folder('../output')
        self.platform = platform
        self.angle_ff_strength = angle_ff_strength
        self.le_distance = le_distance
        self.le_ff_strength = le_ff_strength
        self.ev_ff_strength = ev_ff_strength
        self.ev_ff_power = ev_ff_power
        self.tolerance = tolerance
    
    def run_pipeline(self, run_MD=True, friction=0.1, integrator_step=10 * mm.unit.femtosecond, sim_step=100, ff_path='forcefields/classic_sm_ff.xml', init_struct='rw', temperature=310, p_ev=0, plots=False, continuous_topoisomerase=False):
        '''
        This is the basic function that runs the molecular simulation pipeline.
        '''
        # Parameters
        self.p_ev = p_ev
        self.continuous_topoisomerase = continuous_topoisomerase

        # Define initial structure
        print('Building initial structure...')
        points = compute_init_struct(self.N_beads, mode='rw')
        write_mmcif(points, self.path+'/LE_init_struct.cif')
        generate_psf(self.N_beads, self.path+'/other/LE_init_struct.psf')
        print('Done brother ;D\n')

        # Define System
        pdb = PDBxFile(self.path+'/LE_init_struct.cif')
        forcefield = ForceField(ff_path)
        self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1*u.nanometer)
        integrator = mm.LangevinIntegrator(temperature, friction, integrator_step)

        # Add forces
        print('Adding forces...')
        self.add_forcefield()
        print('Forces added ;)\n')

        # Minimize energy
        print('Minimizing energy...')
        platform = mm.Platform.getPlatformByName(self.platform)
        self.simulation = Simulation(pdb.topology, self.system, integrator, platform)
        self.simulation.reporters.append(StateDataReporter(stdout, (self.N_steps*sim_step)//100, step=True, totalEnergy=True, potentialEnergy=True, temperature=True))
        self.simulation.reporters.append(DCDReporter(self.path+'/other/stochastic_LE.dcd', sim_step))
        self.simulation.context.setPositions(pdb.positions)
        current_platform = self.simulation.context.getPlatform()
        print(f"Simulation will run on platform: {current_platform.getName()}")
        self.simulation.minimizeEnergy(tolerance=self.tolerance)
        print('Energy minimization done :D\n')

        # Run molecular dynamics simulation
        if run_MD:
            print('Running molecular dynamics (wait for 100 steps)...')
            start = time.time()
            heats = list()
            for i in range(self.N_steps):
                # Define probabilities or regions that EV would be disabled
                if self.continuous_topoisomerase:
                    region_length = max(1, int(self.p_ev * self.N_beads))  # Determine region length from p_ev
                    start_idx = np.random.randint(0, self.N_beads - region_length -1)
                    end_idx = start_idx + region_length
                    self.ps_ev = np.zeros(self.N_beads)
                    self.ps_ev[start_idx:end_idx] = 1
                elif p_ev > 0:
                    self.ps_ev = np.random.rand(self.N_beads)
                self.change_loop(i)
                self.change_ev()
                self.simulation.step(sim_step)
                self.state = self.simulation.context.getState(getPositions=True)
                PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.path+f'/ensemble/MDLE_{i+1}.cif', 'w'))
                heats.append(get_heatmap(self.state.getPositions(), save=False))
            end = time.time()
            elapsed = end - start
            print(f'Everything is done! Simulation finished successfully!\nMD finished in {elapsed/60:.2f} minutes.\n')

            self.avg_heat = np.average(heats, axis=0)
            self.std_heat = np.std(heats, axis=0)
            
            if plots:
                np.save(self.path+f'/other/avg_heatmap.npy', self.avg_heat)
                np.save(self.path+f'/other/std_heatmap.npy', self.std_heat)
                self.plot_heat(self.avg_heat, f'/plots/avg_heatmap.svg')
                self.plot_heat(self.std_heat, f'/plots/std_heatmap.svg')
        return self.avg_heat

    def change_ev(self):
        ev_strength = (self.ps_ev > self.p_ev).astype(int) * np.sqrt(self.ev_ff_strength) if self.p_ev > 0 else np.sqrt(self.ev_ff_strength) * np.ones(self.N_beads)
        for n in range(self.N_beads):
            self.ev_force.setParticleParameters(n, [ev_strength[n], 0.05])
        self.ev_force.updateParametersInContext(self.simulation.context)
    
    def change_loop(self,i):
        force_idx = self.system.getNumForces()-1
        self.system.removeForce(force_idx)
        self.add_loops(i)
        self.simulation.context.reinitialize(preserveState=True)
        self.LE_force.updateParametersInContext(self.simulation.context)

    def add_evforce(self):
        'Leonard-Jones potential for excluded volume'
        self.ev_force = mm.CustomNonbondedForce(f'(epsilon1*epsilon2*(sigma1*sigma2)/(r+r_small))^{self.ev_ff_power}')
        self.ev_force.addGlobalParameter('r_small', defaultValue=0.1)
        self.ev_force.addPerParticleParameter('sigma')
        self.ev_force.addPerParticleParameter('epsilon')
        for i in range(self.N_beads):
            self.ev_force.addParticle([np.sqrt(self.ev_ff_strength),0.05])
        self.system.addForce(self.ev_force)

    def add_bonds(self):
        'Harmonic bond borce between succesive beads'
        self.bond_force = mm.HarmonicBondForce()
        for i in range(self.N_beads - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 3e5)
        self.system.addForce(self.bond_force)
    
    def add_stiffness(self):
        'Harmonic angle force between successive beads so as to make chromatin rigid'
        self.angle_force = mm.HarmonicAngleForce()
        for i in range(self.N_beads - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, self.angle_ff_strength)
        self.system.addForce(self.angle_force)
    
    def add_loops(self,i=0):
        'LE force that connects cohesin restraints'
        self.LE_force = mm.HarmonicBondForce()
        for nn in range(self.N_coh):
            self.LE_force.addBond(self.M[nn,i], self.N[nn,i], self.le_distance, self.le_ff_strength)
        self.system.addForce(self.LE_force)

    def add_forcefield(self):
        '''
        Here is the definition of the forcefield.

        There are the following energies:
        - ev force: repelling LJ-like forcefield
        - harmonic bond force: to connect adjacent beads.
        - angle force: for polymer stiffness.
        - loop forces: this is a list of force objects. Each object corresponds to a different cohesin. It is needed to define a force for each time step.
        '''
        self.add_evforce()
        self.add_bonds()
        self.add_stiffness()
        self.add_loops()

    def plot_heat(self,img,file_name):
        figure(figsize=(10, 10))
        plt.imshow(img,cmap="Reds",vmax=1)
        plt.savefig(self.path+file_name,format='svg',dpi=500)
        plt.close()

def main():
    # A potential example
    M = np.load('/home/skorsak/Projects/mine/RepliSage/output/other/Ms.npy')
    N = np.load('/home/skorsak/Projects/mine/RepliSage/output/other/Ns.npy')
    md = MD_LE(M,N,np.max(N)+1,platform='CUDA')
    md.run_pipeline()