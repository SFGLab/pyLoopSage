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

class EM_LE:
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
    
    def run_pipeline(self,plots=False, friction=0.1, ff_path=default_xml_path, integrator_step = 100 * mm.unit.femtosecond, temperature = 310, init_struct='rw'):
        '''
        This is the basic function that runs the molecular simulation pipeline.
        '''
        # Define initial structure
        print('Building initial structure...')
        points = compute_init_struct(self.N_beads,mode=init_struct)
        write_mmcif(points,self.path+'/LE_init_struct.cif')
        generate_psf(self.N_beads,self.path+'/other/LE_init_struct.psf')
        print('Done brother ;D\n')

        counter = 0
        sum_heat = np.zeros((self.N_beads,self.N_beads))
        print('Running energy minimizations...')
        for i in tqdm(range(self.burnin,self.N_steps)):
            # Define System
            pdb = PDBxFile(self.path+'/LE_init_struct.cif')
            forcefield = ForceField(ff_path)
            self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1*u.nanometer)
            integrator = mm.LangevinIntegrator(temperature, friction, integrator_step)

            # Add forces
            ms,ns = self.M[:,i], self.N[:,i]
            self.add_forcefield(ms,ns)

            # Minimize energy
            platform = mm.Platform.getPlatformByName(self.platform)
            simulation = Simulation(pdb.topology, self.system, integrator, platform)
            simulation.context.setPositions(pdb.positions)
            simulation.minimizeEnergy(tolerance=self.tolerance)
            self.state = simulation.context.getState(getPositions=True)
            PDBxFile.writeFile(pdb.topology, self.state.getPositions(), open(self.path+f'/ensemble/EMLE_{i-self.burnin+1}.cif', 'w'))
            sum_heat+=get_heatmap(self.state.getPositions(),save=False)
            counter+=1
        print('Energy minimizations done :D\n')

        self.avg_heat = sum_heat/counter
        
        if plots:
            figure(figsize=(10, 10))
            plt.imshow(self.avg_heat,cmap="Reds",vmax=1)
            plt.colorbar()
            np.save(self.path+f'/other/avg_heatmap.npy',self.avg_heat)
            plt.savefig(self.path+f'/plots/avg_heatmap.svg',format='svg',dpi=500)
            plt.savefig(self.path+f'/plots/avg_heatmap.pdf',format='pdf',dpi=500)
            # plt.colorbar()
            plt.close()
            
        return self.avg_heat

    def add_forcefield(self,ms,ns):
        '''
        Here is the definition of the forcefield.

        There are the following energies:
        - ev force: repelling LJ-like forcefield
        - harmonic bond force: to connect adjacent beads.
        - angle force: for polymer stiffness.
        - LE forces: this is a list of force objects. Each object corresponds to a different cohesin. It is needed to define a force for each time step.
        '''
        # Leonard-Jones potential for excluded volume
        self.ev_force = mm.CustomNonbondedForce(f'epsilon*((sigma1+sigma2)/r)^{self.ev_ff_power}')
        self.ev_force.addGlobalParameter('epsilon', defaultValue=self.ev_ff_strength)
        self.ev_force.addPerParticleParameter('sigma')
        self.system.addForce(self.ev_force)
        for i in range(self.system.getNumParticles()):
            self.ev_force.addParticle([0.05])

        # Harmonic bond borce between succesive beads
        self.bond_force = mm.HarmonicBondForce()
        self.system.addForce(self.bond_force)
        for i in range(self.system.getNumParticles() - 1):
            self.bond_force.addBond(i, i + 1, 0.1, 300000.0)

        # Harmonic angle force between successive beads so as to make chromatin rigid
        self.angle_force = mm.HarmonicAngleForce()
        self.system.addForce(self.angle_force)
        for i in range(self.system.getNumParticles() - 2):
            self.angle_force.addAngle(i, i + 1, i + 2, np.pi, self.angle_ff_strength)
        
        # LE force that connects cohesin restraints
        self.LE_force = mm.HarmonicBondForce()
        self.system.addForce(self.LE_force)
        for nn in range(self.N_coh):
            self.LE_force.addBond(ms[nn], ns[nn], self.le_distance, self.le_ff_strength)
        
def main():
    # A potential example
    M = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ms.npy')
    N = np.load('/home/skorsak/Dropbox/LoopSage/files/region_[48100000,48700000]_chr3/Annealing_Nbeads500_ncoh50/Ns.npy')
    md = EM_LE(4*M,4*N,2000,5,1)
    md.run_pipeline()
