# pyLoopSage
Updated version of the stochastic loop extrusion model: LoopSage with capability to run incredibly fast, parallelized across CPU cores. This package is even more user-friendly and it can be installed via PyPI.

## New features

- More user-friendly environment.
- Installable with `pip install pyLoopSage`.
- Parralelization of the stochastic simulation across multiple CPUs.
- Capability of modelling the whole chromosome.
- Visualization functions.
- Capability to run form terminal with a simple command `loopsage -c config.ini`.

## Publication
Please cite the method paper in case that you would like to use this model for your work,

* Korsak, Sevastianos, and Dariusz Plewczynski. "LoopSage: An energy-based Monte Carlo approach for the loop extrusion modelling of chromatin." Methods (2024).

## The model

We have a polymer chain with $N_{\text{beads}}$ number of monomers. In general we can scale by deault the granularity of the simulation so as to give reasonable results. Therefore if we have a `region` counted in genomic coordinates, we can assume that `N_beads=(region[1]-region[0])//2000`.

Let's assume that each cohesin $i$ can be represented of two coordinates $(m_{i},n_{i})$ we allow three moves in our simulation:

* Slide both locations randomly (as 1D random walk) or
* Rebind somewhere else.

In general a good working assumption is that the number of cohesins (or loop extrusion factors LEFs) is $N_{\text{lef}}=2N_{\text{CTCF}}$.

The main idea of the algorithm is to ensemble loop extrusion from a Boltzmann probability distribution, with Hamiltonian,

$$E = c_{\text{fold}}\sum_{i=1}^{N_{\text{coh}}}\log(n_i-m_i)+c_{\text{cross}}\sum_{i,j}K(m_i,n_i;m_j,n_j)+c_{\text{bind}}\sum_{i=1}^{N_{\text{coh}}}\left(L(m_i)+R(n_i)\right)$$

The first term corresponds to the folding of chromatin, and the second term is a penalty for the appearance of crosss. Therefore, we have the function,
$K(m_{i},n_{i};m_{j},n_{j})$ which takes the value 1 when $m_{i} < m_{j} < n_{i} < n_{j}$ or $m_{i}=m_{j}$ or $m_{i}=n_{j}$.

These $L(\cdot), R(\cdot)$ functions are two functions that define the binding potential and they are orientation specific - so they are different for left and right position of cohesin (because CTCF motifs are orientation specific), therefore when we have a gap in these functions, it means presence of CTCF. These two functions are derived from data with CTCF binning and by running the script for probabilistic orientation. Moreover, by $N_{(\cdot)}$ we symbolize the normalization constants for each factor,

$$c_{\text{fold}}=-\dfrac{N_{\text{beads}}f}{N_{\text{lef}}\log(N_{\text{beads}}/N_{\text{lef}})},\quad c_{\text{bind}}=-\dfrac{N_{\text{beads}}b}{\sum_i \left(L(m_i)+R(n_i)\right)},\quad c_{\text{cross}}=\kappa \times 10^4.$$

The parameters are defined in such a way that when $f=b=\kappa=1$, the three terms of the stochastic energy are balanced. 

And the energy difference can be expressed as the energy difference of each term,

$$\Delta E = \Delta E_{\text{fold}}+\Delta E_{\text{cross}}+\Delta E_{\text{bind}}.$$

In this manner we accept a move in two cases:

* If $\Delta E<0$ or,
* if $\Delta E>0$ with probability $e^{-\Delta E/kT}$.

And of course, the result - the distribution of loops in equilibrium -  depends on temperature of Boltzmann distribution $T$.

## Installation

Can be easily installed with `pip install pyLoopSage`. To have CUDA acceleration, it is needed to have cuda-toolkit installed in case that you use nvidia drivers (otherwise you can use OpenCL or parallelization across CPU cores).

## How to use?

### Python Implementation

The main script is `LoopSage.py`. The implementation of the code is very easy and it can be described in the following lines,

```python
# Definition of Monte Carlo parameters
import loopsage.stochastic_simulation as lps

N_steps, MC_step, burnin, T, T_min = int(4e4), int(5e2), 1000, 2.5, 1.0
mode = 'Metropolis'

# Simulation Strengths
f, b, kappa = 1.0, 1.0, 1.0

# Definition of region
region, chrom = [15550000,16850000], 'chr6'

# Definition of data
output_name='../HiChIP_Annealing_T1_MD_region'
bedpe_file = '/home/skorsak/Data/HiChIP/Maps/hg00731_smc1_maps_2.bedpe'

sim = lps.StochasticSimulation(region,chrom,bedpe_file,out_dir=output_name,N_beads=1000)
Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,mode=mode,viz=True,save=True)
sim.run_MD('CUDA')
```

Firstly, we need to define the input files from which LoopSage would take the information to construct the potential. We define also the specific region that we would like to model. Therefore, in the code script above we define a `bedpe_file` from which information about the CTCF loops it is imported.

Note that the `.bedpe_file` must be in the following format,

```
chr1	903917	906857	chr1	937535	939471	16	3.2197903072213415e-05	0.9431392038374097
chr1	979970	987923	chr1	1000339	1005916	56	0.00010385804708107556	0.9755733944997329
chr1	980444	982098	chr1	1063024	1065328	12	0.15405319074060866	0.999801529750033
chr1	981076	985322	chr1	1030933	1034105	36	9.916593137693526e-05	0.01617512105347667
chr1	982171	985182	chr1	990837	995510	27	2.7536240913152036e-05	0.5549511180231224
chr1	982867	987410	chr1	1061124	1066833	71	1.105408615726611e-05	0.9995462969421808
chr1	983923	985322	chr1	1017610	1019841	11	1.7716275555648395e-06	0.10890923034907056
chr1	984250	986141	chr1	1013038	1015474	14	1.7716282101935205e-06	0.025665007111095667
chr1	990949	994698	chr1	1001076	1003483	34	0.5386388489931403	0.9942742844900859
chr1	991375	993240	chr1	1062647	1064919	15	1.0	0.9997541297643132
```

where the last two columns represent the probabilites for left and right anchor respectively to be tandem right. If the probability is negative it means that no CTCF motif was detected in this anchor. You can extract these probabilities from the repo: https://github.com/SFGLab/3d-analysis-toolkit, with `find_motifs.py` file. 

Then, we define the main parameters of the simulation `N_beads,N_coh,kappa,f,b` or we can choose the default ones (take care because it might be the case that they are not the appropriate ones and they need to be changed), the parameters of Monte Carlo `N_steps, MC_step, burnin, T`, and we initialize the class `LoopSage()`. The command `sim.run_energy_minimization()` corresponds to the stochastic Monte Carlo simulation, and it produces a set of cohesin constraints as a result (`Ms, Ns`). Note that the stochastic simulation has two modes: `Annealing` and `Metropolis`. We feed cohesin constraints to the molecular simulation part of and we run `MD_LE()` or `EM_LE()` simulation which produces a trajectory of 3d-structures, and the average heatmap. `MD_LE()` function can produce an actual trajectory and a `.dcd` video of how simulation changes over time. However, the implementation needs a high amount of memory since we need to define a bond for each time step, and it may not work for large systems. `EM_LE()` is suggested for large simulations, because it does not require so big amount of memory. *We would like to advise to our users that in case that they would like to use molecular dynamics (`MD`), it is better to choose small `MC_step`, because the trajectory of LEFs should be continuous.*

### Running LoopSage from command-line
To run LoopSage from command-line, you only need to type a command

```bash
loopsage -c config.ini
```

With this command the model will run with parameters specified in `config.ini` file. An example of a `config.ini` file would be the following,

```txt
[Main]

; Input Data and Information
BEDPE_PATH = /home/skorsak/Data/HiChIP/Maps/hg00731_smc1_maps_2.bedpe
REGION_START = 15550000
REGION_END = 16850000
CHROM = chr6
OUT_PATH = ../HiChIP_Annealing_T15_MD_region

; Simulation Parameters
N_BEADS = 1000
N_STEPS = 40000
MC_STEP = 500
BURNIN = 1000
T_INIT = 1.5
T_FINAL = 0.01
METHOD = Metropolis

; Molecular Dynamics
PLATFORM = CUDA
INITIAL_STRUCTURE_TYPE = rw
SIMULATION_TYPE = EM 
TOLERANCE = 1.0
``` 

### Visualization with PyVista

There are many tools for visualization of polymer structures. A very good one is UCSF chimera: https://www.cgl.ucsf.edu/chimera/. Usually these visualization tools work well for proteins, but we can use them for chromatin as well.

LoopSage offers its own visualization which relies in the `pyvista` python library. To visualize a structure, you can run some very simple commands, which call LoopSage functions,

```python
import loopsage.vizualization_tools as vz
import loopsage.utils as uts

V = uts.get_coordinates_cif('/home/skorsak/Projects/mine/LoopSage/HiChIP_Annealing_T15_EM_region/ensemble/EMLE_1.cif')
vz.viz_structure(V)
```

The output should be something like that,

![image](https://github.com/user-attachments/assets/457d2b1f-e037-4ff1-8cec-c9eef7de789d)


In case that you would like to create a continuous video from the enseble of structures, you can use the following command, which would generate an interactive video in gif format which would show how the structure changes in 3D. The command includes quaternion Kabsch aligment as well.

```python
import loopsage.vizualization_tools as vz

vz.interactive_plot('/home/skorsak/Projects/mine/LoopSage/HiChIP_Annealing_T15_EM_region/ensemble')
```

### Output Files
In the output files, simulation produces one folder with 4 subfolders. In subfolder `plots`, you can find plots that are the diagnostics of the algorithm. One of the most basic results you should see is the trajectories of cohesins (LEFs). this diagram should look like that,

![coh_trajectories](https://github.com/SFGLab/LoopSage/assets/49608786/f73ffd2b-8359-4c6d-958b-9a770d4834ba)

In this diagram, each LEF is represented by a different colour. In case of Simulated Annealing, LEFs should shape shorter loops in the first simulation steps, since they have higher kinetic energy due to the high temperature, and very stable large loops in the final steps if the final temperature $T_f$ is low enough. Horizontal lines represent the presence of CTCF points. In case of Metropolis, the distribution of LEFs should look invariant in respect to computational time,

![coh_trajectories](https://github.com/SFGLab/LoopSage/assets/49608786/b48e2383-2509-4c68-a726-91a5e61aabf3)

Good cohesin trajectory diagrams should be like the ones previously shown, which means that we do not want to see many unoccupied (white) regions, but we also do not like to see static loops. If the loops are static then it is better to choose higher temperature, or bigger number of LEFs. If the loops are too small, maybe it is better to choose smaller temperature.

Now, to reassure that our algorithm works well we need to observe some fundamental diagnostics of Monte Carlo algorithms. Some other important diagnostics can be seen in the following picture,

![github_diagnostics](https://github.com/SFGLab/LoopSage/assets/49608786/b90e2355-be84-47c4-906f-5a2a62497b26)

In graph A, we can see the plot of the energy as a function of simulation time. In Metropolis after some steps, the simulation should reach equillibrium after the defined burnin period (blue vertical line). In case of Simulated Annealing, the energy should decrease as function of time because the temperature decreases, and thus the energy should not be conserved. Autocorrelations (B), show us if the thermodyncamic ensembles that we obtained are autocorrelated or not. The Monte Carlo step (sampling frequency) should be big enough so as to have small autocorrelations (<0.5). The averaged heatmap (C), shows the simulated heatmap, after averaging inversed all-versus-all distances of the region of interest. Finally, (D) shows the Pearson correlation between projected 1D signal of heatmap of experimental and simulated data.

In the output folder there are another three subfolders: 
* `ensembles` has the ensembles of 3D structures in `.cif` format (it can open with vizualization software Chimera: https://www.cgl.ucsf.edu/chimera/) or `pyvista`,
* `heatmaps` with the inversed all-versus-all distance heatmap of each one of these structures.
* `other` here are some numpy arrays and some computed statistics. Numpy arrays like `Ms` and `Ns` have the degrees of freedoms of LEFs over time, then `Fs, Ks, Es` they have folding, corssing energy and total energy over time. `Ts` is the temperature over time. And finally in `other.txt` you can see the statistics of the simulation and the input parameters. In `correlations.txt` you can find a file with Pearson, Spearmann and Kendau tau correlations between estimated and experimental data. We provide an optimistic simulations where zeros of the signal are taken into account, and a pessimistic one where the zeros are not taken into account.

An example, illustrated with Chimera software, simulated trajectory of structures after running Simulated Annealing and molecular dynamics.

![siman_traj_GitHub](https://github.com/SFGLab/LoopSage/assets/49608786/c6626641-f709-46e0-b01b-42566b1829ef)
