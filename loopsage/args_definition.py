import datetime
import re
from dataclasses import dataclass
from math import pi
from typing import Union
import argparse
import openmm as mm
import importlib.resources
from openmm.unit import Quantity

# Dynamically set the default path to the XML file in the package
try:
    with importlib.resources.path('loopsage.forcefields', 'classic_sm_ff.xml') as default_xml_path:
        default_xml_path = str(default_xml_path)
except FileNotFoundError:
    # If running in a development setup without the resource installed, fallback to a relative path
    default_xml_path = 'loopsage/forcefields/classic_sm_ff.xml'

@dataclass
class Arg(object):
    name: str
    help: str
    type: type
    default: Union[str, float, int, bool, Quantity, None]
    val: Union[str, float, int, bool, Quantity, None]
    nargs: Union[str, None] = None  # Optional attribute for nargs

# Define custom type to parse list from string
def parse_list(s):
    try:
        return [int(x.strip()) for x in s.strip('[]').split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid list format. Must be a comma-separated list of integers.")

class ListOfArgs(list):
    quantity_regexp = re.compile(r'(?P<value>[-+]?\d+(?:\.\d+)?) ?(?P<unit>\w+)')

    def get_arg(self, name: str) -> Arg:
        """Stupid arg search in list of args"""
        name = name.upper()
        for i in self:
            if i.name == name:
                return i
        raise ValueError(f"No such arg: {name}")

    def __getattr__(self, item):
        return self.get_arg(item).val

    def parse_args(self):
        parser = argparse.ArgumentParser()
        for arg in self.arg_list:
            parser.add_argument(arg['name'], help=arg['help'], type=arg.get('type', str), default=arg.get('default', ''), val=arg.get('val', ''))

        args = parser.parse_args()
        parsed_args = {arg['name']: getattr(args, arg['name']) for arg in self.arg_list}
        return parsed_args

    def parse_quantity(self, val: str) -> Union[Quantity, None]:
        if val == '':
            return None
        match_obj = self.quantity_regexp.match(val)
        value, unit = match_obj.groups()
        try:
            unit = getattr(mm.unit, unit)
        except AttributeError:
            raise ValueError(f"I Can't recognise unit {unit} in expression {val}. Example of valid quantity: 12.3 femtosecond.")
        return Quantity(value=float(value), unit=unit)

    def parse_list_of_floats(self, s: str) -> list:
        try:
            return [float(x.strip()) for x in s.strip('[]').split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid list format. Must be a comma-separated list of floats.")

    def parse_list_of_strings(self, s: str) -> list:
        try:
            return [x.strip() for x in s.strip('[]').split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid list format. Must be a comma-separated list of strings.")

    def to_python(self):
        """Casts string args to ints, floats, bool..."""
        for i in self:
            if i.val == '':
                i.val = None
            elif i.name == "HR_K_PARAM":  # Workaround for complex unit
                i.val = Quantity(float(i.val), mm.unit.kilojoule_per_mole / mm.unit.nanometer ** 2)
            elif i.type == str:
                continue
            elif i.type == int:
                i.val = int(i.val)
            elif i.type == float:
                i.val = float(i.val)
            elif i.type == list:
                if i.name == "BW_FILES" or i.name == "STRING_LIST":  # Handle lists of strings
                    if isinstance(i.val, str):
                        i.val = self.parse_list_of_strings(i.val)
                    elif not isinstance(i.val, list):
                        raise ValueError(f"Can't convert {i.val} into list of strings.")
                elif i.name == "BW_STRENGTHS" or i.name == "FLOAT_LIST":  # Handle lists of floats
                    if isinstance(i.val, str):
                        i.val = self.parse_list_of_floats(i.val)
                    elif not isinstance(i.val, list):
                        raise ValueError(f"Can't convert {i.val} into list of floats.")
                else:  # Handle lists of integers
                    if isinstance(i.val, str):
                        i.val = parse_list(i.val)
                    elif not isinstance(i.val, list):
                        raise ValueError(f"Can't convert {i.val} into list of integers.")
            elif i.type == bool:
                if i.val.lower() in ['true', '1', 'y', 'yes']:
                    i.val = True
                elif i.val.lower() in ['false', '0', 'n', 'no']:
                    i.val = False
                else:
                    raise ValueError(f"Can't convert {i.val} into bool type.")
            elif i.type == Quantity:
                try:
                    i.val = self.parse_quantity(i.val)
                except AttributeError:
                    raise ValueError(f"Can't parse: {i.name} = {i.val}")
            else:
                raise ValueError(f"Can't parse: {i.name} = {i.val}")

    def get_complete_config(self) -> str:
        w = "####################\n"
        w += "#   LoopSage Model   #\n"
        w += "####################\n\n"
        w += "# This is automatically generated config file.\n"
        w += f"# Generated at: {datetime.datetime.now().isoformat()}\n\n"
        w += "# Notes:\n"
        w += "# Some fields require units. Units are represented as objects from mm.units module.\n"
        w += "# Simple units are parsed directly. For example: \n"
        w += "# HR_R0_PARAM = 0.2 nanometer\n"
        w += "# But more complex units does not have any more sophisticated parser written, and will fail.'\n"
        w += "# In such cases the unit is fixed (and noted in comment), so please convert complex units manually if needed.\n"
        w += "# <float> and <int> types does not require any unit. Quantity require unit.\n\n"
        w += "# Default values does not mean valid value. In many places it's only a empty field that need to be filled.\n\n"

        w += '[Main]'
        for i in self:
            w += f'; {i.help}, type: {i.type.__name__}, default: {i.default}\n'
            if i.val is None:
                w += f'{i.name} = \n\n'
            else:
                if i.type == Quantity:
                    # noinspection PyProtectedMember
                    w += f'{i.name} = {i.val._value} {i.val.unit.get_name()}\n\n'
                else:
                    w += f'{i.name} = {i.val}\n\n'
        w = w[:-2]
        return w

    def write_config_file(self):
        auto_config_filename = 'config_auto.ini'
        with open(auto_config_filename, 'w') as f:
            f.write(self.get_complete_config())
        print(f"Automatically generated config file saved in {auto_config_filename}")

available_platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]

args = ListOfArgs([
    # Platform settings
    Arg('PLATFORM', help=f"Name of the platform. Available choices: {' '.join(available_platforms)}", type=str, default='CPU', val='CPU'),
    Arg('DEVICE', help="Device index for CUDA or OpenCL (count from 0)", type=str, default='', val=''),
    
    # Input data
    Arg('N_BEADS', help="Number of Simulation Beads.", type=int, default='', val=''),
    Arg('BEDPE_PATH', help="A .bedpe file path with loops. It is required.", type=str, default='', val=''),
    Arg('LEF_TRACK_FILE', help="An optional track file for cohesin or condensin in bw format. If this file is specified LEF preferentially binds were the signal is enriched.", type=str, default='', val=''),
    Arg('BW_FILES', help="List of bigWig file paths for feature extraction.", type=list, nargs='+', default=[], val=[]),
    Arg('OUT_PATH', help="Output folder name.", type=str, default='../results', val='../results'),
    Arg('REGION_START', help="Starting region coordinate.", type=int, default='', val=''),
    Arg('REGION_END', help="Ending region coordinate.", type=int, default='', val=''),
    Arg('CHROM', help="Chromosome that corresponds the the modelling region of interest (in case that you do not want to model the whole genome).", type=str, default='', val=''),
    Arg('FLOAT_LIST', help="List of floating-point numbers.", type=list, nargs='+', default=[], val=[]),
    Arg('STRING_LIST', help="List of strings.", type=list, nargs='+', default=[], val=[]),
    
    # Stochastic Simulation parameters
    Arg('LEF_RW', help="True in case that you would like to make cohesins slide as random walk, instead of sliding only in one direction.", type=bool, default='True', val='True'),
    Arg('LEF_DRIFT', help="True in case that LEFs are pushed back when they encounter other LEFs.", type=bool, default='False', val='False'),
    Arg('N_STEPS', help="Number of Monte Carlo steps.", type=int, default='40000', val='40000'),
    Arg('N_LEF', help="Number of loop extrusion factors (condensins and cohesins). If you leave it empty it would add for LEFs twice the number of CTCFs.", type=int, default='', val=''),
    Arg('N_LEF2', help="Number of second family loop extrusion factors, in case that you would like to simulate a second group with different speed.", type=int, default='0', val='0'),
    Arg('MC_STEP', help="Monte Carlo frequency. It should be hundreds of steps so as to avoid autocorrelated ensembles.", type=int, default='200', val='200'),
    Arg('BURNIN', help="Burnin-period (steps that are considered before equillibrium).", type=int, default='1000', val='1000'),
    Arg('T_INIT', help="Initial Temperature of the Stochastic Model.", type=float, default='2.0', val='2.0'),
    Arg('T_FINAL', help="Final Temperature of the Stochastic Model.", type=float, default='1.0', val='1.0'),
    Arg('METHOD', help="Stochastic modelling method. It can be Metropolis or Simulated Annealing.", type=str, default='Annealing', val='Annealing'),
    Arg('FOLDING_COEFF', help="Folding coefficient.", type=float, default='1.0', val='1.0'),
    Arg('FOLDING_COEFF2', help="Folding coefficient for the second family of LEFs.", type=float, default='0.0', val='0.0'),
    Arg('CROSS_COEFF', help="LEF crossing coefficient.", type=float, default='1.0', val='1.0'),
    Arg('BW_STRENGTHS', help="List of strengths of the energy (floats) corresponding to each BW file. This equivalent to the `r` parameter in the LoopSage paper.", type=list, nargs='+', default=[], val=[]),
    Arg('CROSS_LOOP', help="It true if the penalty is applied for situations mi<mj<ni<nj and mi=nj, and false if it is applied only for mi=nj.", type=bool, default='True', val='True'),
    Arg('BETWEEN_FAMILIES_PENALTY', help="Penalty for LEF2s that are crossing LEFs.", type=bool, default='True', val='True'),
    Arg('BIND_COEFF', help="CTCF binding coefficient.", type=float, default='1.0', val='1.0'),
    Arg('SAVE_PLOTS', help="It should be true in case that you would like to save diagnostic plots. In case that you use small MC_STEP or large N_STEPS is better to mark it as False.", type=bool, default='True', val='True'),
    Arg('SAVE_MDT', help="In case that you would liketo save metadata of the stochastic simulation.", type=bool, default='True', val='True'),
    
    # Molecular Dynamic Properties
    Arg('INITIAL_STRUCTURE_TYPE', help="you can choose between: rw, confined_rw, self_avoiding_rw, helix, circle, spiral, sphere.", type=str, default='rw', val='rw'),
    Arg('SIMULATION_TYPE', help="It can be either EM (multiple energy minimizations) or MD (one energy minimization and then run molecular dynamics).", type=str, default='', val=''),
    Arg('INTEGRATOR_STEP', help="The step of the integrator.", type=Quantity, default='100 femtosecond', val='100 femtosecond'),
    Arg('FORCEFIELD_PATH', help="Path to XML file with forcefield.", type=str, default=default_xml_path, val=default_xml_path),
    Arg('ANGLE_FF_STRENGTH', help="Angle force strength.", type=float, default='200.0', val='200.0'),
    Arg('LE_FF_LENGTH', help="Equillibrium distance of loop forces.", type=float, default='0.1', val='0.1'),
    Arg('LE_FF_STRENGTH', help="Interaction Strength of loop forces.", type=float, default='50000.0', val='50000.0'),
    Arg('CONTINUOUS_TOP', help="True if topoisomerase disables EV in a continuous region rather than a discrete set of points.", type=bool, default='False', val='False'),
    Arg('EV_P', help="Probability that randomly excluded volume may be disabled.", type=float, default='0.0', val='0.0'),
    Arg('EV_FF_STRENGTH', help="Excluded-volume strength.", type=float, default='100.0', val='100.0'),
    Arg('EV_FF_POWER', help="Excluded-volume power.", type=float, default='3.0', val='3.0'),
    Arg('FRICTION',help='Friction coefficient of the Langevin integrator.',type=float, default='0.1', val='0.1'),
    Arg('TOLERANCE', help="Tolerance that works as stopping condition for energy minimization.", type=float, default='0.001', val='0.001'),
    Arg('VIZ_HEATS', help="Visualize the output average heatmap.", type=bool, default='True', val='True'),
    Arg('SIM_TEMP', help="The temperature of the 3D simulation (EM or MD).", type=Quantity, default='310 kelvin', val='310 kelvin'),
    Arg('SIM_STEP', help="This is the amount of simulation steps that are perform each time that we change the loop forces. If this number is too high, the simulation is slow, if is too low it may not have enough time to adapt the structure to the new constraints.", type=int, default='1000', val='1000'),
])