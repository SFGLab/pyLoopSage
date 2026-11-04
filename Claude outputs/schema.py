"""Single source of truth for every parameter.

A Param with a `flag` is also exposed on the command line (the most important
ones); all Params can be set in config.ini under their section.
"""
from dataclasses import dataclass

from common.constants import DEFAULT_FORCEFIELD


@dataclass(frozen=True)
class Param:
    name: str
    section: str
    kind: str            # int | float | bool | str | floats | strs
    default: object
    help: str
    unit: str = ""       # shown in help / report; "fs" and "K" are canonical units
    flag: str = ""       # long CLI flag (empty = ini only)
    short: str = ""
    choices: tuple = ()


SECTIONS = {
    "input": "Input data and genomic region",
    "output": "Output and reproducibility",
    "monte_carlo": "Monte Carlo (loop extrusion) sampler",
    "energy": "Energy model coefficients",
    "molecular_dynamics": "Molecular dynamics (3D structures)",
    "analysis": "Analysis",
}

_INIT = ("rw", "confined_rw", "self_avoiding_rw", "circle", "helix", "spiral", "sphere")

PARAMS = (
    # ---- input ----
    Param("interaction_file", "input", "str", None, "Loops/peaks file: .bedpe, .bed or .narrowPeak (required)", flag="--file", short="-f"),
    Param("chrom", "input", "str", None, "Chromosome, e.g. chr1 (CLI: use --region)"),
    Param("region_start", "input", "int", None, "Region start (empty = chromosome start)", "bp"),
    Param("region_end", "input", "int", None, "Region end (empty = chromosome end)", "bp"),
    Param("genome", "input", "str", "hg38", "Genome assembly for chromosome sizes", flag="--genome", choices=("hg38", "hg19")),
    Param("n_beads", "input", "int", None, "Number of simulation beads (empty = region / 2 kb)", "beads", flag="--n-beads", short="-n"),
    Param("smoothing_input", "input", "bool", False, "Smooth the anchor/peak signal with a Gaussian"),
    Param("bw_files", "input", "strs", [], "BigWig tracks for extra binding energy (comma-separated)"),
    Param("bw_strengths", "input", "floats", [], "Strength r per BigWig track (empty = -N_beads/10 each)"),
    Param("lef_track_file", "input", "str", None, "BigWig of LEF loading preference (cohesin/condensin ChIP)"),
    Param("comp_bw_file", "input", "str", None, "BigWig of compartments (A/B signal)"),
    Param("comp_bed_file", "input", "str", None, "BED of compartments / subcompartments"),
    # ---- output ----
    Param("out_path", "output", "str", "loopsage_out", "Output folder", flag="--out", short="-o"),
    Param("seed", "output", "int", None, "Random seed (empty = random)", flag="--seed"),
    Param("report", "output", "bool", True, "Write report.html", flag="--report"),
    Param("save_ensemble", "output", "bool", True, "Save every MD frame as .cif", flag="--save-ensemble"),
    # ---- Monte Carlo ----
    Param("n_steps", "monte_carlo", "int", 40000, "Total Monte Carlo steps", "MC steps", flag="--mc-steps"),
    Param("mc_step", "monte_carlo", "int", 200, "Save one snapshot every this many steps", "MC steps", flag="--sample-every"),
    Param("burnin", "monte_carlo", "int", 1000, "Burn-in period discarded from statistics", "MC steps", flag="--burnin"),
    Param("t_init", "monte_carlo", "float", 2.0, "Initial temperature", "a.u.", flag="--t-init"),
    Param("t_final", "monte_carlo", "float", 1.0, "Final temperature (Annealing only)", "a.u.", flag="--t-final"),
    Param("method", "monte_carlo", "str", "Annealing", "Sampling method", flag="--method", choices=("Metropolis", "Annealing")),
    Param("n_lef", "monte_carlo", "int", None, "Loop extrusion factors (empty = N_anchors / 2)", "LEFs", flag="--n-lef"),
    Param("n_lef2", "monte_carlo", "int", 0, "Second LEF family (different speed)", "LEFs"),
    Param("lef_rw", "monte_carlo", "bool", True, "LEF legs slide as random walk (False: only extrude outwards)"),
    Param("variable_sliding_step", "monte_carlo", "bool", False, "Sliding step uniform in [1, shortest loop] beads"),
    Param("p_spin", "monte_carlo", "float", 0.5, "Probability of an epigenetic-state move instead of a LEF move"),
    # ---- energy ----
    Param("folding_coeff", "energy", "float", 1.0, "Folding coefficient f"),
    Param("folding_coeff2", "energy", "float", 0.0, "Folding coefficient f2 of the second family"),
    Param("bind_coeff", "energy", "float", 1.0, "Anchor binding coefficient b"),
    Param("cross_coeff", "energy", "float", 1.0, "LEF crossing coefficient kappa"),
    Param("epi_coeff", "energy", "float", 0.0, "Epigenetic (Potts) coupling; 0 = off (experimental)"),
    Param("n_epi_states", "energy", "int", 3, "Number of Potts epigenetic states"),
    Param("cross_loop", "energy", "bool", True, "Penalise crossing loops, not only shared anchors"),
    Param("between_families_penalty", "energy", "bool", True, "Apply crossing penalty between LEF families"),
    Param("data_loss_mode", "energy", "int", 0, "0 = 1D anchor profiles L/R; 1 = 2D loop map"),
    Param("contrastive_binding", "energy", "bool", False, "Penalise loop-free regions (tanh-contrast signals)"),
    # ---- molecular dynamics ----
    Param("run_md", "molecular_dynamics", "bool", True, "Run the 3D molecular dynamics stage", flag="--run-MD"),
    Param("platform", "molecular_dynamics", "str", "CPU", "OpenMM platform: CPU, CUDA, OpenCL, Reference", flag="--platform"),
    Param("initial_structure_type", "molecular_dynamics", "str", "rw", "Initial polymer shape", flag="--init-structure", choices=_INIT),
    Param("integrator_step", "molecular_dynamics", "float", 100.0, "Integrator time step", "fs", flag="--step-size"),
    Param("sim_step", "molecular_dynamics", "int", 2000, "MD steps per MC snapshot (per frame)", "MD steps", flag="--md-steps"),
    Param("sim_temp", "molecular_dynamics", "float", 310.0, "Simulation temperature", "K", flag="--temperature"),
    Param("friction", "molecular_dynamics", "float", 0.1, "Langevin friction", "1/ps"),
    Param("tolerance", "molecular_dynamics", "float", 0.001, "Energy-minimisation tolerance", "kJ/mol/nm"),
    Param("angle_ff_strength", "molecular_dynamics", "float", 100.0, "Chain stiffness (angle force)", "kJ/mol/rad^2"),
    Param("le_ff_length", "molecular_dynamics", "float", 0.1, "Loop-anchor equilibrium distance", "nm"),
    Param("le_ff_strength", "molecular_dynamics", "float", 60000.0, "Loop-anchor spring constant", "kJ/mol/nm^2"),
    Param("ev_ff_strength", "molecular_dynamics", "float", 100.0, "Excluded-volume strength", "kJ/mol"),
    Param("ev_ff_power", "molecular_dynamics", "float", 3.0, "Excluded-volume exponent"),
    Param("ev_p", "molecular_dynamics", "float", 0.0, "Topoisomerase: probability/fraction with excluded volume off"),
    Param("continuous_top", "molecular_dynamics", "bool", False, "Topoisomerase acts on one continuous stretch"),
    Param("do_compartments", "molecular_dynamics", "bool", False, "Add block-copolymer compartment force (needs epi_coeff > 0)"),
    Param("md_skip_burnin", "molecular_dynamics", "bool", True, "Simulate only post-burn-in MC snapshots"),
    Param("forcefield_path", "molecular_dynamics", "str", DEFAULT_FORCEFIELD, "OpenMM force-field XML"),
    # ---- analysis ----
    Param("contact_radius", "analysis", "float", 1.0, "Contact distance for MD contact maps", "nm"),
    Param("detect_knots", "analysis", "bool", False, "Compute linking numbers of loops (needs topoly)", flag="--knots"),
    Param("knots_stride", "analysis", "int", 10, "Use every n-th MD frame for linking numbers", "frames"),
)

BY_NAME = {p.name: p for p in PARAMS}

# keys from old LoopSage configs that no longer do anything (ignored with a note)
DEPRECATED = {"save_plots", "save_mdt", "viz_heats", "device", "float_list", "string_list",
              "lef_drift", "simulation_type", "auto_tune", "comp_file"}
