from .stochastic_simulation import *
from .args_definition import *
import os
import time
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

def get_config() -> ListOfArgs:
    """This function prepares the list of arguments.
    At first List of args with defaults is read.
    Then it's overwritten by args from config file (ini file).
    In the end config is overwritten by argparse options."""

    print(f"Reading config...")
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-c', '--config_file', help="Specify config file (ini format)", metavar="FILE")
    for arg in args:
        arg_parser.add_argument(f"--{arg.name.lower()}", help=arg.help)
    args_ap = arg_parser.parse_args()  # args from argparse
    config_parser = configparser.ConfigParser()
    config_parser.read(args_ap.config_file)
    args_cp = my_config_parser(config_parser)
    # Override defaults args with values from config file
    for cp_arg in args_cp:
        name, value = cp_arg
        arg = args.get_arg(name)
        arg.val = value
    # Now again override args with values from command line.
    for ap_arg in args_ap.__dict__:
        if ap_arg not in ['config_file']:
            name, value = ap_arg, getattr(args_ap, ap_arg)
            if value is not None:
                arg = args.get_arg(name)
                arg.val = value
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
    
    # Simulation Strengths
    f, f2, b, kappa = args.FOLDING_COEFF,  args.FOLDING_COEFF2, args.BIND_COEFF, args.CROSS_COEFF
    
    # Definition of region
    region, chrom = [args.REGION_START,args.REGION_END], args.CHROM
    
    # Definition of data
    output_name = args.OUT_PATH
    bedpe_file = args.BEDPE_PATH
    
    # Run Simulation
    sim = StochasticSimulation(region,chrom,bedpe_file,out_dir=output_name,N_beads=N_beads,N_lef=N_lef,N_lef2=N_lef2)
    Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,mode=mode,viz=args.SAVE_PLOTS,save=args.SAVE_MDT,lef_rw=args.LEF_RW,f=f,f2=f2,b=b,kappa=kappa)
    if args.SIMULATION_TYPE=='EM':
        sim.run_EM(args.PLATFORM,args.ANGLE_FF_STRENGTH,args.LE_FF_LENGTH,args.LE_FF_STRENGTH,args.EV_FF_STRENGTH,args.EV_FF_POWER,args.TOLERANCE,args.FRICTION,args.INTEGRATOR_STEP,args.SIM_TEMP,args.VIZ_HEATS,args.FORCEFIELD_PATH)
    elif args.SIMULATION_TYPE=='MD':
        sim.run_MD(args.PLATFORM,args.ANGLE_FF_STRENGTH,args.LE_FF_LENGTH,args.LE_FF_STRENGTH,args.EV_FF_STRENGTH,args.EV_FF_POWER,args.TOLERANCE,args.FRICTION,args.INTEGRATOR_STEP,args.SIM_TEMP,args.SIM_STEP,args.VIZ_HEATS,args.FORCEFIELD_PATH)
    elif args.SIMULATION_TYPE==None:
        print('\n3D simulation did not run because it was not specified. Please specify argument SIMULATION_TYPE as EM or MD.')
    else:
        IndentationError('Uknown simulation type. It can be either MD or EM.')

if __name__=='__main__':
    main()