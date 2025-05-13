from .stochastic_simulation import *
from .args_definition import *
from .knots import *
import argparse
import configparser
from typing import List
from sys import stdout

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
    """Prepare list of arguments.
    First, defaults are set.
    Then, optionally config file values.
    Finally, CLI arguments overwrite everything."""
    
    print("Reading config...")

    # Step 1: Setup argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--config_file', help="Specify config file (ini format)", metavar="FILE")

    for arg in args:
        arg_parser.add_argument(f"--{arg.name.lower()}", help=arg.help)
    
    args_ap = arg_parser.parse_args()  # parse command-line arguments
    args_dict = vars(args_ap)

    # Step 2: If config file provided, parse it
    if args_ap.config_file:
        config_parser = configparser.ConfigParser()
        config_parser.read(args_ap.config_file)
        args_cp = my_config_parser(config_parser)

        # Override default args with values from config file
        for cp_arg in args_cp:
            name, value = cp_arg
            arg = args.get_arg(name)
            arg.val = value

    # Step 3: Override again with CLI arguments (if present)
    for name, value in args_dict.items():
        if name == "config_file":
            continue
        if value is not None:
            arg = args.get_arg(name.upper())
            arg.val = value

    # Step 4: Finalize
    args.to_python()
    args.write_config_file()
    
    return args

def main():
    # Input arguments
    args = get_config()
    
    # Monte Carlo Parameters
    N_beads, N_lef, N_lef2 = args.N_BEADS, args.N_LEF, args.N_LEF2
    N_steps, MC_step, burnin, T, T_min = args.N_STEPS, args.MC_STEP, args.BURNIN, args.T_INIT, args.T_FINAL
    mode = args.METHOD
    bw_paths = args.BW_FILES
    
    # Simulation Strengths
    f, f2, b, kappa = args.FOLDING_COEFF,  args.FOLDING_COEFF2, args.BIND_COEFF, args.CROSS_COEFF
    r = args.BW_STRENGTHS
    between_families_penalty = args.BETWEEN_FAMILIES_PENALTY  # Added argument
    
    # Definition of region
    region, chrom = [args.REGION_START,args.REGION_END], args.CHROM
    
    # Definition of data
    output_name = args.OUT_PATH
    bedpe_file = args.BEDPE_PATH
    
    # Run Simulation
    sim = StochasticSimulation(region,chrom,bedpe_file,out_dir=output_name,N_beads=N_beads,N_lef=N_lef,N_lef2=N_lef2, bw_files=bw_paths, track_file=args.LEF_TRACK_FILE)
    Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,mode=mode,viz=args.SAVE_PLOTS,save=args.SAVE_MDT,lef_rw=args.LEF_RW,f=f,f2=f2,b=b,kappa=kappa,lef_drift=args.LEF_DRIFT,cross_loop=args.CROSS_LOOP,r=r,between_families_penalty=between_families_penalty)
    if args.SIMULATION_TYPE=='EM':
        sim.run_EM(args.PLATFORM,args.ANGLE_FF_STRENGTH,args.LE_FF_LENGTH,args.LE_FF_STRENGTH,args.EV_FF_STRENGTH,args.EV_FF_POWER,args.TOLERANCE,args.FRICTION,args.INTEGRATOR_STEP,args.SIM_TEMP,args.INITIAL_STRUCTURE_TYPE,args.VIZ_HEATS,args.FORCEFIELD_PATH)
    elif args.SIMULATION_TYPE=='MD':
        sim.run_MD(args.PLATFORM,args.ANGLE_FF_STRENGTH,args.LE_FF_LENGTH,args.LE_FF_STRENGTH,args.EV_FF_STRENGTH,args.EV_FF_POWER,args.TOLERANCE,args.FRICTION,args.INTEGRATOR_STEP,args.SIM_TEMP,args.INITIAL_STRUCTURE_TYPE,args.SIM_STEP,args.VIZ_HEATS,args.FORCEFIELD_PATH,args.EV_P,args.CONTINUOUS_TOP)
    elif args.SIMULATION_TYPE==None:
        print('\n3D simulation did not run because it was not specified. Please specify argument SIMULATION_TYPE as EM or MD.')
    else:
        IndentationError('Uknown simulation type. It can be either MD or EM.')

    # Knoting
    if args.DETECT_KNOTS:
        link_number_ensemble(path=args.OUT_PATH, viz=args.SAVE_PLOTS, mode=args.SIMULATION_TYPE)
    
if __name__=='__main__':
    main()