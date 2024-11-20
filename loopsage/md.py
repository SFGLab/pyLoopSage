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
import importlib.resources
from .utils import *
from .initial_structures import *

# Dynamically set the default path to the XML file in the package
try:
    with importlib.resources.path('loopsage.forcefields', 'classic_sm_ff.xml') as default_xml_path:
        default_xml_path = str(default_xml_path)
except FileNotFoundError:
    # If running in a development setup without the resource installed, fallback to a relative path
    default_xml_path = 'loopsage/forcefields/classic_sm_ff.xml'

class MD_LE:
    def __init__(self,M,N,N_beads,burnin,MC_step,path=None,platform='CPU',angle_ff_strength=200,le_distance=0.1,le_ff_strength=50000.0,ev_ff_strength=10.0,ev_ff_power=3.0,tolerance=0.001):
        '''
        M, N (np arrays): Position matrix of two legs of cohesin m,n. 
                          Rows represent  loops/cohesins and columns represent time
        N_beads (int): The number of beads of initial structure.
        step (int): sampling rate
        path (int): the path where the simulation will save structures etc.
        '''
        self.M, self.N = M, N
        self.N_coh, self.N_steps = M.shape
        self.N_beads, self.step, self.burnin = N_beads, MC_step, burnin//MC_step
        self.path = path
        self.platform = platform
        self.angle_ff_strength = angle_ff_strength
        self.le_distance = le_distance
        self.le_ff_strength = le_ff_strength
        self.ev_ff_strength = ev_ff_strength
        self.ev_ff_power = ev_ff_power
        self.tolerance = tolerance
    
    def run_pipeline(self,run_MD=True, friction=0.1, integrator_step=100 * mm.unit.femtosecond, sim_step=1000, ff_path=default_xml_path, temperature=310, plots=False):
        '''
        This is the basic function that runs the molecular simulation pipeline.
        '''
        # Parameters
        self.angle_ff_strength=200
        self.le_distance=0.1
        self.le_ff_strength=300000.0
        self.tolerance=0.001

        # Define initial structure
        print('Building initial structure...')
        points = compute_init_struct(self.N_beads,mode='rw')
        write_mmcif(points,self.path+'/LE_init_struct.cif')
        generate_psf(self.N_beads,self.path+'/other/LE_init_struct.psf')
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
            for i in range(1,self.N_steps):
                self.change_loop(i)
                self.simulation.step(sim_step)
                if i>=self.burnin:
                    self.state = self.simulation.context.getState(getPositions=True)
                    PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.path+f'/ensemble/MDLE_{i-self.burnin+1}.cif', 'w'))
                    heats.append(get_heatmap(self.state.getPositions(),save=False))
            end = time.time()
            elapsed = end - start

            print(f'Everything is done! Simulation finished succesfully!\nMD finished in {elapsed/60:.2f} minutes.\n')

            self.avg_heat = np.average(heats,axis=0)
            self.std_heat = np.std(heats,axis=0)
            
            if plots:
                np.save(self.path+f'/other/avg_heatmap.npy',self.avg_heat)
                np.save(self.path+f'/other/std_heatmap.npy',self.std_heat)
                self.plot_heat(self.avg_heat,f'/plots/avg_heatmap.svg')
                self.plot_heat(self.std_heat,f'/plots/std_heatmap.svg')
        return self.avg_heat
    
    def change_loop(self,i):
        force_idx = self.system.getNumForces()-1
        self.system.removeForce(force_idx)
        self.add_loops(i)
        self.simulation.context.reinitialize(preserveState=True)
        self.LE_force.updateParametersInContext(self.simulation.context)

    def add_evforce(self):
        'Leonard-Jones potential for excluded volume'
        self.ev_force = mm.CustomNonbondedForce(f'epsilon*((sigma1+sigma2)/(r+r_small))^{self.ev_ff_power}')
        self.ev_force.addGlobalParameter('epsilon', defaultValue=self.ev_ff_strength)
        self.ev_force.addGlobalParameter('r_small', defaultValue=0.01)
        self.ev_force.addPerParticleParameter('sigma')
        for i in range(self.N_beads):
            self.ev_force.addParticle([0.05])
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
    M = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ms.npy')
    N = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ns.npy')
    md = MD_LE(4*M,4*N,2000,5,1)
    md.run_pipeline()
