# Molecular Dynamics

The [Monte Carlo](monte-carlo.md) stage tells us *which beads are held together by loops* in each snapshot. It says nothing about where the beads are in space. The molecular dynamics (MD) stage turns each snapshot into a 3D structure: it builds a polymer, ties the loop anchors together with springs, and lets the chain relax and move. The set of structures of all snapshots is the 3D ensemble.

The stage uses **OpenMM**, a molecular simulation library. It is optional (`--run-MD`).

## The idea

Think of the chromatin fibre as a string of $N$ beads. In each frame:

1. the string is a chain of beads held by bonds, and it resists bending and passing through itself;
2. the loops of the Monte Carlo snapshot are added as springs between the two anchor beads of each loop;
3. the system is simulated for a number of MD steps, so the chain can find a shape that agrees with all springs;
4. the final positions are saved, and the next frame starts from them with the loops of the next snapshot.

Step 4 matters. The simulation is *not* restarted for every frame. It continues from the previous structure, and only the springs are exchanged. Neighbouring snapshots have almost the same loops, so the structure changes gradually and a few thousand steps per frame are enough to relax it. The number of steps per frame, `sim_step`, is therefore the *relaxation time* between two sets of loops: a larger number gives the chain more time to adapt, a smaller one gives faster but more correlated structures.

!!! note "Units"
    Lengths are in nanometres and energies in kJ/mol, but the model is coarse-grained: a bead is a few kb of chromatin and the bond length is 0.1 nm in the force field. The numbers are model units, not real nanometres of a nucleus, and the time of a frame is a model time, not a measured one. Only ratios, contact patterns and shapes are meant to be compared with data.

## The forces

The force field is a sum of simple terms. They are all *potential energies*: the simulation moves the beads to lower the total.

| Force | What it does | Form | Option |
|---|---|---|---|
| **Bonds** | keeps neighbouring beads at distance $r_0=0.1$ nm | $\tfrac12 k\,(r-r_0)^2$, $k=3\cdot10^{5}$ kJ/mol/nm² (fixed) | |
| **Angles** | stiffness: penalises bending away from a straight line | $\tfrac12 k_\theta(\theta-\pi)^2$ | `angle_ff_strength` |
| **Excluded volume** | beads repel when they come close, so the chain cannot pass through itself | a soft power law, see below | `ev_ff_strength`, `ev_ff_power` |
| **Loop springs** | pulls the two anchors of every loop together | $\tfrac12 k_L(r-r_L)^2$ | `le_ff_length` ($r_L$), `le_ff_strength` ($k_L$) |
| **Compartments** (optional) | beads in the same epigenetic state attract each other | a Gaussian well | `do_compartments`, `comp_strength`, `comp_range` |

**Excluded volume.** The repulsion between two beads at distance $r$ is

$$
U_{\mathrm{ev}}(r)=\left(\frac{\varepsilon^2\,\sigma^2}{r+r_s}\right)^{p},\qquad \sigma^2=\texttt{ev\_ff\_strength},\ \ \varepsilon=0.05,\ \ r_s=0.1\ \mathrm{nm},\ \ p=\texttt{ev\_ff\_power}.
$$

It is large when two beads overlap and decays quickly with distance; the exponent $p$ decides how steep: a larger $p$ gives a harder, shorter-ranged wall, and a smaller $p$ a softer, longer-ranged one (the default is $p=6$: a hard core at contact, but a weaker push at a distance than $p=3$, so the chain is not over-avoiding; at the default strength the crossing barrier is about $15\,k_BT$, against $5\,k_BT$ for $p=3$). The softening $r_s$ keeps the force finite when beads touch. How *hard* the wall must be is a real question: if it is too soft, two strands can pass through each other, which creates unphysical knots and links; see [Topology](../theory/topology.md), where the same force is used to set a barrier of a chosen height.

**Loop springs.** The springs are the only link between the Monte Carlo and the 3D structure. They are *replaced* in every frame: the old springs are removed and the springs of the new snapshot added. A stiff spring ($k_L=6\cdot10^4$) and a short rest length ($r_L=0.1$ nm) mean that the two anchors of a loop are practically in contact.

**Compartments.** When the epigenetic model was on in the Monte Carlo (`epi_coeff` > 0) and you set `do_compartments = True`, every bead carries the state it had in the snapshot, and two beads in the same state attract each other,

$$
U_{\mathrm{comp}}(r)=s\,E_{ab}\,e^{-r^2/2w^2},\qquad w=\texttt{comp\_range}\cdot\sqrt N\ \mathrm{nm}.
$$

$E_{ab}$ is a small table: it is non-zero only for two beads of the same state, and it is largest for state 0 (B) and weaker for the others, so that B beads aggregate more strongly than A beads. The scale $s$ is `comp_strength`: 1 is the full strength of the table, lower values give softer compartments that stay open instead of collapsing into one dense ball. The states are updated in every frame, so compartments follow the Monte Carlo.

## The integrator

An *integrator* is the rule that moves the beads from one time step to the next. LoopSage uses one: the **Langevin integrator** of OpenMM. It solves Newton's equation of motion for every bead, with two extra ingredients that mimic the surrounding fluid: a *friction* force that slows the bead down, and *random kicks* that keep it moving. Friction and kicks are tied together so that the chain reaches the temperature you set, and the structures it samples are those of a polymer in thermal equilibrium with the springs of the current loops.

There are no other integrators to choose from, but the Langevin integrator has three numbers that you set:

| Option | Default | Meaning |
|---|---|---|
| `integrator_step` (`--step-size`) | 100 fs | the *time step*: how far the dynamics advances in one iteration |
| `sim_temp` (`--temperature`) | 310 K | the temperature of the heat bath; it sets the size of the random kicks |
| `friction` | 0.1 / ps | how strongly the beads are slowed down; a larger value gives slower, more *overdamped* motion |

A larger time step makes the simulation faster, but too large a step makes it unstable. If the coordinates become infinite the run stops with the message that the MD became unstable; the cure is a smaller `--step-size`, or softer forces (a smaller `le_ff_strength` or `ev_ff_strength`).

## The steps of the stage

**Build.** The initial chain is made in a chosen shape (`initial_structure_type`: `rw` random walk, `confined_rw`, `self_avoiding_rw`, `circle`, `helix`, `spiral` or `sphere`), the forces are added, and the energy is *minimised* once (the beads are moved downhill until the force is below `tolerance`), so that the first frame does not start from overlapping beads.

**Platform.** OpenMM can run on a GPU or on CPU cores. With `platform = auto` LoopSage tries CUDA, then OpenCL, then the CPU; if a platform is listed but cannot start (a driver problem), it moves on to the next one. On the CPU the number of threads is the `--threads` value (all cores if you gave none). `device` picks a GPU when there are several.

**Frames.** For every Monte Carlo snapshot (after the burn-in when `md_skip_burnin` is on, otherwise all of them): swap the loop springs and the compartment states, run `sim_step` integrator steps, then record the positions, the radius of gyration, the potential energy, the distance matrix and the contact map. Each structure is also written as a `.cif` file when `save_ensemble` is on.

The simulated time of one frame is `sim_step` $\times$ `integrator_step` (with the defaults $5000\times100$ fs $=500$ ps). The default of 5000 steps per frame was chosen so that the chain relaxes between two sets of loops.

**Contact maps.** A contact map is built from the distances $d_{ij}$ of every frame with a Gaussian kernel $e^{-d_{ij}^2/2\rho^2}$, where $\rho$ is `contact_radius` (default 1 nm), and averaged over the frames. Its comparison with Hi-C is part of the [Analysis](pipeline.md) stage.

## Topoisomerase options

Real chromatin is not a polymer that can never cross itself: topoisomerases let strands pass through each other. Two optional modes switch the excluded volume off for a short time, so that strands can cross (`topo_mode = random` with `ev_p`, or `topo_mode = local` with `topo_rate`, `topo_radius`, `topo_width`, `topo_window`, `topo_keep`). Whether the chain crosses itself without help can be counted with `track_crossings`. This is the subject of the [Topology](../theory/topology.md) page.

## The arguments

```ini
[molecular_dynamics]
run_md       = True          # run this stage                                  (CLI: --run-MD)
platform     = auto          # auto, CUDA, OpenCL, CPU or Reference            (CLI: --platform)
device       =               # GPU index, empty = default

initial_structure_type = rw  # rw, confined_rw, self_avoiding_rw, circle, helix, spiral, sphere
                             #                                                 (CLI: --init-structure)
# integrator (Langevin)
integrator_step = 100.0      # time step, fs                                   (CLI: --step-size)
sim_step        = 5000       # MD steps per Monte Carlo snapshot (frame)       (CLI: --md-steps)
sim_temp        = 310.0      # temperature, K                                  (CLI: --temperature)
friction        = 0.1        # 1/ps
tolerance       = 0.001      # energy minimisation, kJ/mol/nm

# forces
angle_ff_strength = 50.0     # stiffness, kJ/mol/rad^2
ev_ff_strength    = 100.0    # excluded volume strength, kJ/mol
ev_ff_power       = 6.0      # excluded volume exponent
le_ff_length      = 0.1      # loop spring rest length, nm
le_ff_strength    = 60000.0  # loop spring constant, kJ/mol/nm^2

# compartments (needs epi_coeff > 0 in [energy])
do_compartments = False
comp_strength   = 0.25       # scale of the interaction table
comp_range      = 0.03       # range in units of sqrt(N) nm

md_skip_burnin  = False      # only use snapshots after the Monte Carlo burn-in
save_ensemble   = True       # write the structures (frames as .cif, trajectory)  (CLI: --save-ensemble)

[analysis]
contact_radius  = 1.0        # nm, for the MD contact maps
```

!!! tip "Typical changes"
    For a more flexible, more compact chain lower `angle_ff_strength`; for a stiffer, more extended one raise it. For stronger aggregation of A and B compartments raise `comp_strength`. To let the structure relax longer between two sets of loops raise `--md-steps`. To save time while testing, lower `--md-steps` and save fewer snapshots (a larger `--sample-every`).

## Where the code is

| File | Role |
|---|---|
| `src/molecular_dynamics/engine.py` | the OpenMM engine: platform choice, build, the loop over frames, the contact maps |
| `src/molecular_dynamics/forces.py` | the force terms: bonds, angles, excluded volume, loop springs, compartments |
| `src/molecular_dynamics/structures.py` | the shapes of the starting chain |
| `src/molecular_dynamics/topology.py` | the topoisomerase passage events and the crossing counter |
| `src/pipeline/stage_md.py` | the pipeline stage: runs the engine, writes tables and figures |

### Outputs

The coordinates of all frames, the radius of gyration, the energies and the average contact and distance maps are saved in `data/md.npz`. The folder `structures/` contains the initial structure (`init_structure.cif`), the trajectory (`trajectory.dcd`) and, with `save_ensemble`, one file `MDLE_<k>.cif` per frame in `structures/ensemble/`. The 3D stage then builds the interactive viewer and the videos from these frames.
