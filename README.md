# pyLoopSage
Updated version of the stochastic loop extrusion model: LoopSage with capability to run incredibly fast, parallelized across CPU cores. This package is even more user-friendly and it can be installed via PyPI.

## New features

- More user-friendly environment.
- Installable with `pip install pyLoopSage`.
- Parralelization of the stochastic simulation across multiple CPUs.
- Capability of modelling the whole chromosome.
- Visualization functions.
- Capability to run form terminal with a simple command `loopsage -c config.ini`.

## The model

### Stochastic Simulation
Before simulating 3D structures, a stochastic simulation takes place that it purely applies in the locations of cohesins. The purpose of it is to create realistic ensemble of cohesin locations, and recontruct their trajectories. Otherwise, a molecular simulation alone, could not have enough variability to reconstruct the experimental heatmaps with the same fixed positions. Therefore, the stochastic simulation can be seen as a generator of cohesin possible configurations.

We have a polymer chain with $N_{\text{beads}}$ number of monomers. In general we can scale by deault the granularity of the simulation so as to give reasonable results. Therefore if we have a `region` counted in genomic coordinates, we can assume that `N_beads=(region[1]-region[0])//2000`.

Let's assume that each cohesin $i$ can be represented of two coordinates $(m_{i},n_{i})$ we allow three moves in our simulation:

* Slide both locations randomly (linearly or as 1D random walk) or
* Rebind somewhere else.

In general a good working assumption is that the number of cohesins (or loop extrusion factors LEFs) is $N_{\text{lef}}=2N_{\text{CTCF}}$.

The main idea of the algorithm is to ensemble loop extrusion from a Boltzmann probability distribution, with Hamiltonian,

$$E = c_{\text{fold}}\sum_{i=1}^{N_{\text{coh}}}\log(n_i-m_i)+c_{\text{cross}}\sum_{i,j}K(m_i,n_i;m_j,n_j)+c_{\text{bind}}\sum_{i=1}^{N_{\text{coh}}}\left(L(m_i)+R(n_i)\right)$$

The first term corresponds to the folding of chromatin, and the second term is a penalty for the appearance of crosss. Therefore, we have the function,
$K(m_{i},n_{i};m_{j},n_{j})$ which takes the value 1 when $m_{i} < m_{j} < n_{i} < n_{j}$ or $m_{i}=m_{j}$ or $m_{i}=n_{j}$.

These $L(\cdot), R(\cdot)$ functions are two functions that define the binding potential and they are orientation specific - so they are different for left and right position of cohesin (because CTCF motifs are orientation specific), therefore when we have a gap in these functions, it means presence of CTCF. These two functions are derived from data with CTCF binning and by running the script for probabilistic orientation. Moreover, by $N_{(\cdot)}$ we symbolize the normalization constants for each factor,

$$c_{\text{fold}}=-\dfrac{N_{\text{beads}}f}{N_{\text{lef}}\log(N_{\text{beads}}/N_{\text{lef}})},\quad c_{\text{bind}}=-\dfrac{N_{\text{beads}}b}{\sum_i \left(L(m_i)+R(n_i)\right)},\quad c_{\text{cross}}=\kappa \times 10^4.$$

The parameters are defined in such a way that when $f=b=\kappa=1$, the three terms of the stochastic energy are balanced. To clarrify, this parametrization is slightly different than this of the original paper. We multiply with $N_{\text{beads}}$, so as to keep the input parameters $f,b,\kappa$ simpler close to 1. Therefore, a value $f=0.1$ leads to unfolded structures, and a value $f=2$ it leads to very fast propagating long loops. 

And the energy difference can be expressed as the energy difference of each term,

$$\Delta E = \Delta E_{\text{fold}}+\Delta E_{\text{cross}}+\Delta E_{\text{bind}}.$$

In this manner we accept a move in two cases:

* If $\Delta E<0$ or,
* if $\Delta E>0$ with probability $e^{-\Delta E/kT}$.

And of course, the result - the distribution of loops in equilibrium -  depends on temperature of Boltzmann distribution $T$.

#### Modelling with two families of cohesins

In this version of LoopSage it is possible to have two families of cohesins (LEFs) as well, with different folding coefficients (which is equivalent to different cohesin speeds). Therefore, we could write the energy of folding,

$$E_{\text{fold}} = c_{\text{fold,1}}\sum_{i=1}^{N_{\text{coh,1}}}\log(n_i-m_i)+c_{\text{fold,2}}\sum_{i=N_{\text{coh,1}}}^{N_{\text{coh,1}}+N_{\text{coh},2}}\log(n_i-m_i)$$

The coefficient of the first family of LEFs is set to one by default $f_1 = 1$, whereas the other one is set to zero $f_2 = 0$. Usually a small amount of fast cohesins with $f_2>f_1$ can reconstruct more long-range loops, because the cohesins have time to extrude. Contrary, leads in very stable small loops.

### Molecular Simulation

Let us consider a system comprising $N_{\text{lef}}$ LEFs, as well as two matrices, $M$ and $N$, both of which possess dimensions $N_{\text{lef}}\times N_{\text{steps}}$. These matrices represent the respective constraints associated with each LEF. Consequently, we define a time-dependent force field as follows:

$$E(t) = E_{\text{bond}}+E_{\text{angle}}+E_{\text{rep}}+E_{\text{loop}}(t)$$

where,

- $E_{\text{bond}}$ corresponds to a typical harmonic bond force field that connects adjacent beads $i$ and $i+1$, with an equilibrium length of $x_{0}=0.1 \text{nm}$ and a Hook constant assumed to be $k=3\times 10^5 \text{kJ/(mol}\cdot \text{nm}^2)$.
- $E_{\text{angle}}$ a harmonic angle force that connects beads $i-1,i,i+1$, and has equilibrium angle $\theta_{0}=\pi$ and Hook constant $200  kJ/(mol\cdot nm^2)$. The strength of the angle force, can be tuned by the user.
- $E_{\text{rep}}$ which is a repelling forcefield of the form: $$E_{\text{rep}} = \epsilon\left(\frac{\sigma_{1}+\sigma_{2}}{r}\right)^{\alpha}$$ where $\epsilon=10 kJ/mol$ and $\sigma_{1}=\sigma_{2}=0.05 nm$. The power $\alpha$ is a parameter of the simulation, but it is set as $\alpha=3$ by default.
- $E_{\text{loop}}(t)$ represents a time-dependent LE force. This force reads the matrices $M$ and $N$, applying a distinct set of constraints $C_{t_i}=(m_j(t_i),n_j(t_i))$ at each time step $t_i$. Each LEF $j$ is subjected to specific constraints $m_{j,t_i}$ and $n_{j,t_i}$. The functional form of this force is also a harmonic bond force, with parameters $x_{0}=0.1 \text{nm}$ and a Hook constant assumed to be $k=5\times 10^4 \text{kJ/(mol}\cdot \text{nm}^2)$. The strength and equillibrium distance of the looping bonds can also be set in different way, if the user wishes.
  
For the implementation of this model in python, we used OpenMM and CUDA acceleration. To minimize the energy Langevin dynamics were used, in temperature of $T_{L}=310 K$, friction coefficient $\gamma = 0.05  psec^{-1}$ and time step $t_{s}=100 fsec$. Note that the temperature of molecular dynamics simulation is independent from the temperature of stochastic simulation and they represent different physical realities. 

In general the user can run simulation in two different ways:

1. **Energy minimization (EM)**: It means that for each sample of cohesin positions $C_{t_i}=(m_j(t_i),n_j(t_i))$ start from a different initial structure (usually 3D random walk) and we apply the forcefield. For each structure we start from a different initial condition. In general, it is suggested to run the model in this way because it is faster, less prone to errors and the structures are not correlated to each other.
2. **Molecular Dynamics (MD)**: In this case we have only one initial structure, we minimize the energy according to the forcefield only once and then we run a molecular dynamics simulation over time. This creates a continuous trajectory of structures, and it is cool for visualization pruposes. It is also biophysically more correct, in the sense that loop extrusion should be time-dependent, and the structure at time $t_i$ has to me correlated with structures at time $t_{i\pm1}$. It is a little bit more prone to error, and you may need to change the simulation frequence and step in case of instability (smaller frequency and more steps to stabilize it). 

## Installation

Can be easily installed with `pip install pyLoopSage`. To have CUDA acceleration, it is needed to have cuda-toolkit installed in case that you use nvidia drivers (otherwise you can use OpenCL or parallelization across CPU cores).

## How to use?

### Python Implementation

The main script is `stochastic_simulation.py`. However, the implementation of the code is very easy and it can be described in the following lines,

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
sim.run_EM('CUDA')
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

where the last two columns represent the probabilites for left and right anchor respectively to be tandem right. If the probability is negative it means that no CTCF motif was detected in this anchor. You can extract these probabilities from the repo: https://github.com/SFGLab/3d-analysis-toolkit, with `find_motifs.py` file. Please set `probabilistic=True` `statistics=False`.


To use this script you need to have a python 3.8 environment, and install the following dependencies from this file: [mateusz_script_env.txt](https://github.com/user-attachments/files/17568248/mateusz_script_env.txt).

You also need to download the reference genome from: ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa

*Alternativelly, it is possible to import a .bedpe file without the last two columns (CTCF orientation). In this case, CTCF would act as an orientation independent barrier. This might affect slightly the results, but it is an easier option, if you do not want to run a CTCF motif finding script.*

Then, we define the main parameters of the simulation `N_beads,N_coh,kappa,f,b` or we can choose the default ones (take care because it might be the case that they are not the appropriate ones and they need to be changed), the parameters of Monte Carlo `N_steps, MC_step, burnin, T`, and we initialize the class `LoopSage()`. The command `sim.run_energy_minimization()` corresponds to the stochastic Monte Carlo simulation, and it produces a set of cohesin constraints as a result (`Ms, Ns`). Note that the stochastic simulation has two modes: `Annealing` and `Metropolis`. We feed cohesin constraints to the molecular simulation part of and we run `EM_LE()` or `EM_LE()` simulation which produces a trajectory of 3d-structures, and the average heatmap. `MD_LE()` function can produce an actual trajectory and a `.dcd` video of how simulation changes over time. However, the implementation needs a high amount of memory since we need to define a bond for each time step, and it may not work for large systems. `EM_LE()` is suggested for large simulations, because it does not require so big amount of memory.

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

### Long-table of LoopSage arguments

#### General Settings
| Argument Name          | Description                                                                                                     | Type        | Default Value       |
|------------------------|-----------------------------------------------------------------------------------------------------------------|------------|---------------------|
| PLATFORM              | Name of the platform. Available choices: CPU, CUDA, OpenCL                                                      | str        | CPU               |
| DEVICE                | Device index for CUDA or OpenCL (count from 0)                                                                  | str        | None                  |
| OUT_PATH              | Output folder name.                                                                                             | str        | ../results       |
| SAVE_MDT             | True to save metadata of the stochastic simulation.                                                            | bool       | True                |

#### Input Data
| Argument Name          | Description                                                                                                     | Type        | Default Value       |
|------------------------|-----------------------------------------------------------------------------------------------------------------|------------|---------------------|
| BEDPE_PATH            | A .bedpe file path with loops. It is required.                                                                  | str        | None                  |
| LEF_TRACK_FILE        | Path to a bw file of cohesin or condensin density. If it is provided, then simulation LEFs preferentially bind in enriched regions. | str | None |
| BW_FILES              | List of paths to .bw files containing additional data for simulation. Enriched regions would act as barriers for cohesin.                | list       | None                  |
| REGION_START          | Starting region coordinate.                                                                                     | int        | None                  |
| REGION_END            | Ending region coordinate.                                                                                       | int        | None                  |
| CHROM                 | Chromosome that corresponds to the modeling region of interest.                                                 | str        | None                  |

#### Stochastic Simulation Parameters
| Argument Name          | Description                                                                                                     | Type        | Default Value       |
|------------------------|-----------------------------------------------------------------------------------------------------------------|------------|---------------------|
| N_BEADS               | Number of Simulation Beads.                                                                                     | int        | None                  |
| N_STEPS               | Number of Monte Carlo steps.                                                                                    | int        | 40000               |
| MC_STEP               | Monte Carlo frequency to avoid autocorrelated ensembles.                                                       | int        | 200                 |
| BURNIN               | Burn-in period (steps before equilibrium).                                                                      | int        | 1000                |
| T_INIT                | Initial Temperature of the Stochastic Model.                                                                   | float      | 2.0                 |
| T_FINAL               | Final Temperature of the Stochastic Model.                                                                     | float      | 1.0                 |
| METHOD                | Stochastic modeling method (Metropolis or Simulated Annealing).                                                | str        | 'Annealing'         |
| LEF_RW                | True if cohesins slide as a random walk instead of one direction.                                               | bool       | True                |
| LEF_DRIFT             | True if LEFs are pushed back when they encounter other LEFs.                                                   | bool       | False               |
| N_LEF                 | Number of loop extrusion factors.                                                                               | int        | None                  |
| N_LEF2                | Number of second family loop extrusion factors.                                                                | int        | 0                   |
| CROSS_LOOP           | True if the penalty is applied when mi<mj<ni<nj. False if it applies only when mj=ni. When false it is better to enable LEF_DRIFT as well. | bool |True                |
| BETWEEN_FAMILIES_PENALTY | Penalty applied when loops from different LEF families cross.                                                | bool     | True                 |

#### Energy Coefficients
| Argument Name          | Description                                                                                                     | Type        | Default Value       |
|------------------------|-----------------------------------------------------------------------------------------------------------------|------------|---------------------|
| FOLDING_COEFF         | Folding coefficient.                                                                                           | float      | 1.0                 |
| FOLDING_COEFF2        | Folding coefficient for the second family of LEFs.                                                            | float      | 0.0                 |
| CROSS_COEFF          | LEF crossing coefficient.                                                                                      | float      | 1.0                 |
| BIND_COEFF           | CTCF binding coefficient.                                                                                      | float      | 1.0                 |
| BW_STRENGTHS          | List of strengths of the energy (floats) corresponding to each BW file. This equivalent to the `r` parameter in the LoopSage paper.                                                              | list       | None                  |

#### Molecular Dynamics Simulation
| Argument Name          | Description                                                                                                     | Type        | Default Value       |
|------------------------|-----------------------------------------------------------------------------------------------------------------|------------|---------------------|
| SIMULATION_TYPE       | Either EM (energy minimizations) or MD (molecular dynamics).                                                  | str        | None                  |
| INITIAL_STRUCTURE_TYPE | Choose from: rw, confined_rw, self_avoiding_rw, helix, circle, spiral, sphere.                                | str        | rw                |
| INTEGRATOR_STEP       | Step of the integrator.                                                                                        | Quantity   | 100 femtosecond   |
| FORCEFIELD_PATH       | Path to XML file with forcefield.                                                                             | str        | default_xml_path    |
| ANGLE_FF_STRENGTH     | Angle force strength.                                                                                          | float      | 200.0               |
| LE_FF_LENGTH         | Equilibrium distance of loop forces.                                                                          | float      | 0.1                 |
| LE_FF_STRENGTH       | Interaction Strength of loop forces.                                                                          | float      | 50000.0             |
| CONTINUOUS_TOP       | True if the topological constraints are applied continuously during the simulation.                           | bool       | False               |
| EV_P                 | Probability that excluded volume is disabled. Enable it only in case of topoisomerase activity simulation.    | float      | 0.0                 |
| EV_FF_STRENGTH       | Excluded-volume strength.                                                                                     | float      | 100.0               |
| EV_FF_POWER          | Excluded-volume power.                                                                                        | float      | 3.0                 |
| FRICTION             | Friction coefficient of the Langevin integrator.                                                              | float      | 0.1              |
| TOLERANCE            | Stopping condition for energy minimization.     | float      | 0.001              |
| SIM_TEMP             | Temperature of the 3D simulation (EM or MD).                                                                  | Quantity   | 310 kelvin        |
| SIM_STEP             | Amount of simulation steps for loop force adjustments.                                                        | int        | 1000                |

#### Visualization
| Argument Name          | Description                                                                                                     | Type        | Default Value       |
|------------------------|-----------------------------------------------------------------------------------------------------------------|------------|---------------------|
| VIZ_HEATS            | True to visualize the output average heatmap.                                                                | bool       | True                |
| SAVE_PLOTS           | True to save diagnostic plots.                                                                                 | bool       | True                |

## Citation
Please cite the method and biological paper in case that you would like to use this model for your work,

* Korsak, Sevastianos, and Dariusz Plewczynski. "LoopSage: An energy-based Monte Carlo approach for the loop extrusion modelling of chromatin." Methods (2024).
* Jodkowska, K., Parteka-Tojek, Z., Agarwal, A., Denkiewicz, M., Korsak, S., Chiliński, M., Banecki, K., & Plewczynski, D. (2024). Improved cohesin HiChIP protocol and bioinformatic analysis for robust detection of chromatin loops and stripes. In bioRxiv (p. 2024.05.16.594268). https://doi.org/10.1101/2024.05.16.594268
