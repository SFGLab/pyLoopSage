from .stochastic_simulation import *
from .args_definition import *
from .knots import *
from .utils import human_chromosome_lengths
import argparse
import configparser
from typing import List
from sys import stdout
from .logger import get_logger

log = get_logger(__name__)

def my_config_parser(config_parser: configparser.ConfigParser) -> List[tuple[str, str]]:
    """Helper function that makes flat list arg name, and it's value from ConfigParser object."""
    sections = config_parser.sections()
    all_nested_fields = [dict(config_parser[s]) for s in sections]
    args_cp = []
    for section_fields in all_nested_fields:
        for name, value in section_fields.items():
            args_cp.append((name, value))
    return args_cp

def get_config():
    """
    Prepare configuration.

    Priority:
        1. Default values
        2. Configuration file (optional)
        3. Command-line arguments
    """

    log.info("Reading configuration...")

    # Step 1: Parse command-line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--config_file",
        help="Specify configuration file (.ini)",
        metavar="FILE",
    )

    for arg in args:
        arg_parser.add_argument(
            f"--{arg.name.lower()}",
            help=arg.help,
        )

    args_ap = arg_parser.parse_args()
    args_dict = vars(args_ap)

    # Step 2: Read configuration file
    if args_ap.config_file:

        log.info(f"Loading configuration file: {args_ap.config_file}")

        config_parser = configparser.ConfigParser()

        if not config_parser.read(args_ap.config_file):
            log.error(f"Configuration file not found: {args_ap.config_file}")
            raise FileNotFoundError(args_ap.config_file)

        args_cp = my_config_parser(config_parser)

        for name, value in args_cp:
            args.get_arg(name).val = value

        log.info(f"Loaded {len(args_cp)} parameters from configuration file.")

    else:
        log.info("No configuration file specified. Using default parameters.")

    # Step 3: Override with CLI arguments
    n_overwritten = 0

    for name, value in args_dict.items():

        if name == "config_file":
            continue

        if value is not None:
            args.get_arg(name.upper()).val = value
            n_overwritten += 1

    if n_overwritten:
        log.info(f"Overrode {n_overwritten} parameter(s) from the command line.")

    # Step 4: Finalize
    args.to_python()
    args.write_config_file()

    log.info("Configuration successfully initialized.")

    return args

def main():
    # Input arguments
    args = get_config()
    
    # Monte Carlo Parameters
    N_beads, N_lef, N_lef2 = args.N_BEADS, args.N_LEF, args.N_LEF2
    N_steps, MC_step, burnin, T, T_min = args.N_STEPS, args.MC_STEP, args.BURNIN, args.T_INIT, args.T_FINAL
    mode = args.METHOD
    bw_paths = args.BW_FILES
    
    # Compartment inputs (NEW)
    comp_bw_file = getattr(args, "COMP_BW_FILE", "") or None
    comp_bed_file = getattr(args, "COMP_BED_FILE", "") or None

    # optional: enforce exclusivity if both provided
    if comp_bw_file and comp_bed_file:
        raise ValueError("Provide only one compartment file: BW or BED, not both.")

    comp_file = comp_bw_file if comp_bw_file else comp_bed_file
    
    # Simulation Strengths
    f, f2, b, kappa, epi_coeff = args.FOLDING_COEFF,  args.FOLDING_COEFF2, args.BIND_COEFF, args.CROSS_COEFF, args.EPI_COEFF
    r = args.BW_STRENGTHS
    between_families_penalty = args.BETWEEN_FAMILIES_PENALTY  # Added argument
    
    # Definition of genomic region
    try:
        region_start = int(args.Region_start)
        region_end = int(args.Region_end)

        if region_start < 0 or region_end <= region_start:
            raise ValueError

        log.info(
            f"Using genomic region: {args.CHROM}:{region_start:,}-{region_end:,}"
        )

    except (TypeError, ValueError):

        genome = args.GENOME

        try:
            region_start = 0
            region_end = human_chromosome_lengths[genome][args.CHROM]

            log.warning(
                f"Invalid or missing region coordinates. "
                f"Falling back to the entire chromosome "
                f"({args.CHROM}: {region_start:,}-{region_end:,})."
            )

        except KeyError:

            log.error(
                f"Unknown genome/chromosome combination: "
                f"{genome} / {args.CHROM}"
            )

            raise ValueError(
                f"Cannot determine chromosome length for "
                f"genome='{genome}', chromosome='{args.CHROM}'. "
                "Please specify valid --Region_start and --Region_end."
            )

    chrom, region = args.CHROM, [region_start, region_end]
    
    # Definition of data
    output_name = args.OUT_PATH
    bedpe_file = args.BEDPE_PATH
    
    # Run stochastic simulation
    sim = StochasticSimulation(
        region=region,
        chrom=chrom,
        bedpe_file=bedpe_file,
        out_dir=output_name,
        N_beads=N_beads,
        N_lef=N_lef,
        N_lef2=N_lef2,
        bw_files=bw_paths,
        lef_density_file=args.LEF_TRACK_FILE,
        comp_file=comp_file,
        data_loss_mode=args.DATA_LOSS_MODE,
        contrastive_binding=args.CONTRASTIVE_BINDING
    )

    Es, Ms, Ns, Bs, Ks, Fs, ufs, spins = sim.run_energy_minimization(
        N_steps,
        MC_step,
        burnin,
        T,
        T_min,
        mode=mode,
        N_epi_states=args.N_EPI_STATES,
        epi_coeff=args.EPI_COEFF,
        viz=args.SAVE_PLOTS,
        save=args.SAVE_MDT,
        lef_rw=args.LEF_RW,
        lef_drift=args.LEF_DRIFT,
        cross_loop=args.CROSS_LOOP,
        r=r,
        f=f,
        f2=f2,
        b=b,
        kappa=kappa,
        between_families_penalty=between_families_penalty,
        is_variable_step=args.VARIABLE_SLIDING_STEP
    )

    sim.run_MD(
        args.PLATFORM,
        args.ANGLE_FF_STRENGTH,
        args.LE_FF_LENGTH,
        args.LE_FF_STRENGTH,
        args.EV_FF_STRENGTH,
        args.EV_FF_POWER,
        args.DO_COMPARTMENTS,
        args.TOLERANCE,
        args.FRICTION,
        args.INTEGRATOR_STEP,
        args.SIM_TEMP,
        args.INITIAL_STRUCTURE_TYPE,
        args.SIM_STEP,
        args.VIZ_HEATS,
        args.FORCEFIELD_PATH,
        args.EV_P,
        args.CONTINUOUS_TOP
    )

    # Knoting
    if args.DETECT_KNOTS:
        link_number_ensemble(path=args.OUT_PATH, viz=args.SAVE_PLOTS, mode=args.SIMULATION_TYPE)
    
if __name__=='__main__':
    main()