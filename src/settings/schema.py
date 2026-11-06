"""Single source of truth for every parameter.

A Param with a `flag` is also exposed on the command line (the most important
ones); all Params can be set in config.ini under their section.
"""
from dataclasses import dataclass

from common.genome import DEFAULT_FORCEFIELD


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
    "parallel": "Parallelism and replicas",
    "monte_carlo": "Monte Carlo (loop extrusion) sampler",
    "energy": "Energy model coefficients",
    "molecular_dynamics": "Molecular dynamics (3D structures)",
    "structure_3d": "3D structure analysis and rendering",
    "analysis": "Analysis",
    "optimizer": "Parameter optimizer (--optimize)",
    "topology": "Topology: topoisomerase, chain crossings and linking (optional, advanced)",
}

_INIT = ("rw", "confined_rw", "self_avoiding_rw", "circle", "helix", "spiral", "sphere")

PARAMS = (
    # ---- input ----
    Param("interaction_file", "input", "str", None, "Loops/peaks file: .bedpe, .bed or .narrowPeak (required)", flag="--file", short="-f"),
    Param("hic_file", "input", "str", None, "Optional Hi-C map (.hic, read with hic-straw): reference for the validation and, with --optimize, the target of the optimizer", flag="--hic"),
    Param("hic_norm", "input", "str", "KR", "Hi-C normalisation to read (falls back to SCALE, VC_SQRT, VC, NONE when missing)", flag="--hic-norm"),
    Param("hic_resolution", "input", "int", 0, "Hi-C bin size to read (0 = automatic, closest to the bead size)", "bp", flag="--hic-resolution"),
    Param("chrom", "input", "str", None, "Chromosome, e.g. chr1 (CLI: use --region)"),
    Param("region_start", "input", "int", None, "Region start (empty = chromosome start)", "bp"),
    Param("region_end", "input", "int", None, "Region end (empty = chromosome end)", "bp"),
    Param("genome", "input", "str", "hg38", "Genome assembly for chromosome sizes", flag="--genome", choices=("hg38", "hg19")),
    Param("n_beads", "input", "int", None, "Number of simulation beads (empty = region / 2 kb)", "beads", flag="--n-beads", short="-n"),
    Param("smoothing_input", "input", "bool", True, "Smooth the anchor/peak signal with a Gaussian (sigma = n_beads/300 beads, at least 1.5)"),
    Param("bw_files", "input", "strs", [], "BigWig tracks for extra binding energy (comma-separated)"),
    Param("bw_strengths", "input", "floats", [], "Strength r per BigWig track (empty = -N_beads/10 each)"),
    Param("lef_track_file", "input", "str", None, "BigWig of LEF loading preference (cohesin/condensin ChIP)", flag="--lef-track"),
    Param("comp_bw_file", "input", "str", None, "BigWig of compartments (A/B signal)"),
    Param("comp_bed_file", "input", "str", None, "BED of compartments / subcompartments"),
    Param("min_pet", "input", "int", 0, "BEDPE: keep only loops with at least this many PET counts / score (0 = keep all)", "PET", flag="--min-pet"),
    Param("motif_genome", "input", "str", None, "BEDPE: reference genome FASTA; enables the CTCF motif search that sets anchor orientations", flag="--motif-genome"),
    Param("motif_pfm", "input", "str", None, "Motif PFM file (empty = bundled JASPAR CTCF MA0139.1)", flag="--motif-pfm"),
    Param("motif_pvalue", "input", "float", 1e-5, "Motif hit cutoff: chance of a hit in random sequence (smaller = stricter)", "p-value", flag="--motif-pvalue"),
    Param("motif_pseudocount", "input", "float", 0.8, "Pseudocount added to the PFM counts (avoids impossible bases)"),
    Param("motif_pad", "input", "int", 0, "Extend every anchor by this many bp on both sides before the motif search", "bp"),
    # ---- output ----
    Param("out_path", "output", "str", None, "Results folder (empty = <project>/results); each run gets its own subfolder", flag="--out", short="-o"),
    Param("run_number", "output", "int", None, "Run number in the folder name (empty = next free; standalone stages reuse the latest)", flag="--run"),
    Param("verbosity", "output", "int", 0, "Console detail: 0 main steps + progress bars, 1 + tables, 2 + info messages, 3 + debug (CLI: -v, -vv, -vvv)"),
    Param("quiet", "output", "bool", False, "Print nothing to the console (the log file and the report are still written)", flag="--quiet", short="-q"),
    Param("seed", "output", "int", None, "Random seed (empty = a random one, saved in the resolved config); the same seed repeats the run exactly, MD included only with --threads 1", flag="--seed"),
    Param("report", "output", "bool", True, "Write report.html", flag="--report"),
    Param("save_metadata", "output", "bool", True, "Save map matrices (MC and MD heatmaps, distances, strengths) as .npy in data/metadata", flag="--metadata"),
    Param("save_ensemble", "output", "bool", True, "Save every MD frame as .cif", flag="--save-ensemble"),
    # ---- parallelism ----
    Param("threads", "parallel", "int", 1, "Worker threads for the numba sampler, OpenMM CPU and numba kernels (0 = all cores)", "threads", flag="--threads", short="-t"),
    Param("replicas", "parallel", "int", 1, "Independent Monte Carlo chains (seed, seed+1, ...) run in parallel; the ensemble is pooled", "chains", flag="--n-replicas"),
    # ---- Monte Carlo ----
    Param("n_steps", "monte_carlo", "int", 20000, "Total Monte Carlo steps", "MC steps", flag="--mc-steps"),
    Param("mc_step", "monte_carlo", "int", 200, "Save one snapshot every this many steps", "MC steps", flag="--sample-every"),
    Param("burnin", "monte_carlo", "int", 1000, "Burn-in period discarded from statistics", "MC steps", flag="--burnin"),
    Param("t_init", "monte_carlo", "float", 3.0, "Initial temperature", "a.u.", flag="--t-init"),
    Param("t_final", "monte_carlo", "float", 1.0, "Final temperature (Annealing only)", "a.u.", flag="--t-final"),
    Param("method", "monte_carlo", "str", "Annealing", "Sampling method", flag="--method", choices=("Metropolis", "Annealing")),
    Param("n_lef", "monte_carlo", "int", None, "Loop extrusion factors (empty = N_anchors / 2)", "LEFs", flag="--n-lef"),
    Param("n_lef2", "monte_carlo", "int", 0, "Second LEF family (different speed)", "LEFs"),
    Param("device", "molecular_dynamics", "str", "", "Device index for CUDA or OpenCL (count from 0; empty = default)"),
    Param("lef_drift", "monte_carlo", "bool", False, "Kept for compatibility with the original LoopSage (has no effect there either)"),
    Param("lef_rw", "monte_carlo", "bool", True, "LEF legs slide as a random walk (True, original) or only extrude outward (False)", flag="--lef-rw"),
    Param("lef_speed_dist", "monte_carlo", "str", "constant", "Initial LEF speeds: constant (all equal, original), maxwell (Maxwell-Boltzmann), "
          "lognormal, or two_state (slow and fast LEFs); mean speed is always 1", flag="--lef-speed", choices=("constant", "maxwell", "lognormal", "two_state")),
    Param("lef_speed_spread", "monte_carlo", "float", 0.5, "lognormal only: width (sigma of ln speed)", flag="--lef-speed-spread"),
    Param("lef_fast_fraction", "monte_carlo", "float", 0.2, "two_state only: fraction of fast LEFs", flag="--lef-fast-fraction"),
    Param("lef_fast_ratio", "monte_carlo", "float", 5.0, "two_state only: speed of fast LEFs relative to slow ones", flag="--lef-fast-ratio"),
    Param("variable_sliding_step", "monte_carlo", "bool", False, "Sliding step uniform in [1, shortest loop] beads"),
    Param("p_epigenetic", "monte_carlo", "float", 0.5, "Epigenetic model only: probability that a step changes a bead's epigenetic state instead of moving a LEF"),
    # ---- energy ----
    Param("folding_coeff", "energy", "float", 1.0, "Folding coefficient f"),
    Param("folding_coeff2", "energy", "float", 0.0, "Folding coefficient f2 of the second family"),
    Param("bind_coeff", "energy", "float", 1.0, "Anchor binding coefficient b"),
    Param("cross_coeff", "energy", "float", 1.0, "LEF crossing coefficient kappa"),
    Param("epi_coeff", "energy", "float", 0.0, "Epigenetic-model coupling strength; 0 = off (default: pure loop-extrusion model)"),
    Param("n_epi_states", "energy", "int", 3, "Number of epigenetic states"),
    Param("epigenetic_mode", "energy", "str", "dynamic", "Epigenetic coupling J_ij: 'dynamic' = 1 on the backbone and on current LEF loops, "
          "else 0 (follows the LEFs); 'static' = fixed map from the loop data (advanced, config.ini only)", choices=("dynamic", "static")),
    Param("cross_loop", "energy", "bool", True, "Penalise crossing loops, not only shared anchors"),
    Param("between_families_penalty", "energy", "bool", True, "Apply crossing penalty between LEF families"),
    Param("data_loss_mode", "energy", "int", 0, "0 = 1D anchor profiles L/R; 1 = 2D loop map"),
    Param("loss_sigma", "energy", "float", 2.0, "Gaussian width of the 2D loop map J_loss when data_loss_mode = 1, so LEF legs feel a smooth landscape (advanced, config.ini only)", "beads"),
    Param("binding_mode", "energy", "str", "preferential", "Anchor binding mode: preferential = LEFs are attracted to the anchor profiles (non-negative), loop-free regions cost nothing; "
          "contrastive = in addition, loops are penalised for spanning loop-free regions and rewarded for spanning regions the input loops cover (needs loops)", flag="--binding-mode", choices=("preferential", "contrastive")),
    # ---- molecular dynamics ----
    Param("run_md", "molecular_dynamics", "bool", True, "Run the 3D molecular dynamics stage", flag="--run-MD"),
    Param("platform", "molecular_dynamics", "str", "auto", "OpenMM platform: auto (try CUDA, then OpenCL, then parallel CPU), CUDA, OpenCL, CPU or Reference", flag="--platform"),
    Param("initial_structure_type", "molecular_dynamics", "str", "rw", "Initial polymer shape", flag="--init-structure", choices=_INIT),
    Param("integrator_step", "molecular_dynamics", "float", 100.0, "Integrator time step", "fs", flag="--step-size"),
    Param("sim_step", "molecular_dynamics", "int", 5000, "MD steps per MC snapshot (per frame)", "MD steps", flag="--md-steps"),
    Param("sim_temp", "molecular_dynamics", "float", 310.0, "Simulation temperature", "K", flag="--temperature"),
    Param("friction", "molecular_dynamics", "float", 0.1, "Langevin friction", "1/ps"),
    Param("tolerance", "molecular_dynamics", "float", 0.001, "Energy-minimisation tolerance", "kJ/mol/nm"),
    Param("angle_ff_strength", "molecular_dynamics", "float", 50.0, "Chain stiffness (angle force)", "kJ/mol/rad^2"),
    Param("le_ff_length", "molecular_dynamics", "float", 0.1, "Loop-anchor equilibrium distance", "nm"),
    Param("le_ff_strength", "molecular_dynamics", "float", 60000.0, "Loop-anchor spring constant", "kJ/mol/nm^2"),
    Param("ev_ff_strength", "molecular_dynamics", "float", 100.0, "Excluded-volume strength", "kJ/mol"),
    Param("ev_ff_power", "molecular_dynamics", "float", 6.0, "Excluded-volume exponent (larger = harder, shorter-ranged wall; 6 keeps the chain from passing through itself without making it over-avoiding)"),
    Param("do_compartments", "molecular_dynamics", "bool", False, "Add block-copolymer compartment force (needs epi_coeff > 0)"),
    Param("comp_strength", "molecular_dynamics", "float", 0.25, "Block-copolymer compartment attraction: scale of the interaction table (1 = the original strength; lower = softer, compartments stay open instead of collapsing)"),
    Param("comp_range", "molecular_dynamics", "float", 0.03, "Block-copolymer attraction range in units of sqrt(N) nm (the original value was 0.05; smaller = shorter range)"),
    Param("md_skip_burnin", "molecular_dynamics", "bool", False, "Simulate only post-burn-in MC snapshots (original LoopSage uses every snapshot)"),
    Param("forcefield_path", "molecular_dynamics", "str", DEFAULT_FORCEFIELD, "OpenMM force-field XML"),
    # ---- 3D structure ----
    Param("render_3d", "structure_3d", "bool", True, "Write 3d_viewer.html and videos of the 3D trajectory", flag="--render-3d"),
    Param("make_videos", "structure_3d", "bool", True, "Render mp4 videos (needs pyvista; ffmpeg via imageio-ffmpeg)"),
    Param("colorings", "structure_3d", "strs", ["purple", "rainbow"], "Video colourings: purple, rainbow, compartments (added automatically when compartments are loaded)"),
    Param("representation", "structure_3d", "str", "tube", "Video representation: tube (continuous chain) or beads (beads on a string); the viewer offers both"),
    Param("interp_steps", "structure_3d", "int", 5, "Interpolated frames between two MD frames (smooth trajectories)", "frames"),
    Param("video_fps", "structure_3d", "int", 30, "Video frame rate", "fps"),
    Param("video_size", "structure_3d", "int", 640, "Video width and height", "px"),
    Param("max_video_frames", "structure_3d", "int", 240, "Maximum frames per video", "frames"),
    # ---- analysis ----
    Param("corr_smooth", "analysis", "float", 1.5, "Gaussian smoothing (beads) applied to per-position signals before the extra validation correlations (0 = off)", "beads"),
    Param("track_smooth", "analysis", "float", 7.0, "Gaussian smoothing of the Hi-C insulation score and of the eigenvector (A/B track) used as input and for validation (0 = off)", "beads", flag="--track-smooth"),
    Param("contact_radius", "analysis", "float", 1.0, "Contact distance for MD contact maps", "nm"),
    Param("zoom_snapshots", "analysis", "int", 120, "Snapshots shown in the zoomed kymograph (window near equilibrium)", "snapshots"),
    Param("zoom_start", "analysis", "int", None, "First snapshot of the zoom window (empty = middle of production)", "snapshot"),
    Param("optimize", "optimizer", "bool", False, "Before the run, fit the number of LEFs, folding and binding coefficients with Nelder-Mead on short Monte Carlo runs (n_lef of the first family, f, b and the temperature schedule), then run with them", flag="--optimize"),
    Param("opt_evals", "optimizer", "int", 60, "Maximum number of short Monte Carlo evaluations of the optimizer", "evaluations", flag="--opt-evals"),
    Param("opt_early_stop", "optimizer", "bool", True, "Stop the optimizer early when the loss no longer improves", flag="--opt-early-stop"),
    Param("opt_interactive", "optimizer", "bool", True, "Pressing Enter during the optimizer shows the best parameters so far and asks whether to keep them and stop (only in a terminal)", flag="--opt-interactive"),
    Param("opt_patience", "optimizer", "int", 20, "Early stopping: evaluations without a real improvement (see opt_min_gain) before the optimizer stops", "evaluations", flag="--opt-patience"),
    Param("opt_min_gain", "optimizer", "float", 0.01, "Early stopping: smallest drop of the best loss that counts as an improvement (the loss is about 1-4 at the start)"),
    Param("opt_steps", "optimizer", "int", 4000, "Monte Carlo steps of one optimizer evaluation (each step is a sweep over all LEFs; 25% burn-in, 200 snapshots; the final candidate check runs 3x longer)", "MC steps", flag="--opt-steps"),
    Param("opt_max_beads", "optimizer", "int", 1500, "Largest region the optimizer simulates: bigger regions are optimized on the window of this many beads with most loops, and the number of LEFs is scaled back to the full region", "beads", flag="--opt-max-beads"),

    # ---- topology (optional, advanced: nothing here runs by default) ----
    Param("detect_knots", "topology", "bool", False, "Optional linking analysis of loops in 3D (slow; off by default)", flag="--knots"),
    Param("knots_stride", "topology", "int", 10, "Use every n-th MD frame for linking numbers", "frames"),
    Param("ev_p", "topology", "float", 0.0, "Topoisomerase ('random' mode): probability/fraction of beads with excluded volume off during a frame"),
    Param("continuous_top", "topology", "bool", False, "Topoisomerase ('random' mode) acts on one continuous stretch"),
    Param("topo_mode", "topology", "str", "random", "Topoisomerase model: 'random' (beads with EV off for a whole frame, set by ev_p) or 'local' (short strand-passage events at close contacts)", choices=("random", "local")),
    Param("topo_rate", "topology", "float", 5.0, "'local' mode: strand-passage attempts per MD frame per 1000 beads", "events"),
    Param("topo_radius", "topology", "float", 2.0, "'local' mode: strands closer than this can pass through each other", "bond lengths"),
    Param("topo_width", "topology", "int", 3, "'local' mode: beads of the passing strand that lose their excluded volume", "beads"),
    Param("topo_window", "topology", "float", 0.05, "'local' mode: share of the frame's MD steps with excluded volume off, per attempt", "fraction"),
    Param("topo_keep", "topology", "float", 1.0, "'local' mode: probability to keep a passage that creates a new link between loops (1 = no bias; below 1 the topoisomerase simplifies the topology)"),
    Param("track_crossings", "topology", "bool", False, "Count chain crossings during the MD (checks whether the excluded volume leaks); also on with --knots"),
    Param("track_every", "topology", "int", 100, "MD steps between crossing checks", "MD steps"),
    Param("ev_barrier_target", "topology", "float", 10.0, "Strand-crossing barrier used for the suggested ev_ff_strength in the topology report", "kT"),
    Param("target_smooth", "optimizer", "float", 0.8, "Smoothing of the optimizer target built from the input loops, so that the sparse loops look like a Hi-C map (0 = none)", "beads", flag="--target-smooth"),
    Param("hic_denoise", "optimizer", "float", 1.0, "Denoising of the Hi-C target of the optimizer, to bring out the TAD pattern (0 = none); validation always uses the raw map", "beads", flag="--hic-denoise"),
)

BY_NAME = {p.name: p for p in PARAMS}

# keys from old LoopSage configs that no longer do anything (ignored with a note)
DEPRECATED = {"save_plots", "save_mdt", "viz_heats", "float_list", "string_list",
              "simulation_type", "auto_tune", "comp_file"}
