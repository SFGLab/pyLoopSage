import datetime
import re
from dataclasses import dataclass
from math import pi
from typing import Union
import argparse
import openmm as mm
from openmm.unit import Quantity

@dataclass
class Arg(object):
    name: str
    help: str
    type: type
    default: Union[str, float, int, bool, Quantity, None]
    val: Union[str, float, int, bool, Quantity, None]

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
    Arg('PLATFORM', help=f"name of the platform. Available choices: {' '.join(available_platforms)}", type=str, default='CPU', val='CPU'),
    Arg('DEVICE', help="device index for CUDA or OpenCL (count from 0)", type=str, default='', val=''),
    
    # Input data
    Arg('N_BEADS', help="Number of Simulation Beads.", type=int, default='', val=''),
    Arg('BEDPE_PATH', help="A .bedpe file path with loops. It is required.", type=str, default='', val=''),
    Arg('OUT_PATH', help="Output folder name.", type=str, default='../results', val='../results'),
    Arg('REGION_START', help="Starting region coordinate.", type=int, default='', val=''),
    Arg('REGION_END', help="Ending region coordinate.", type=int, default='', val=''),
    Arg('CHROM', help="Chromosome that corresponds the the modelling region of interest (in case that you do not want to model the whole genome).", type=str, default='', val=''),
    
    # Stochastic Simulation parameters
    Arg('N_STEPS', help="Number of Monte Carlo steps.", type=int, default='', val=''),
    Arg('MC_STEP', help="Monte Carlo frequency.", type=int, default='200', val='200'),
    Arg('BURNIN', help="Burnin-period (steps that are considered before equillibrium).", type=int, default='1000', val='1000'),
    Arg('T_INIT', help="Initial Temperature of the Stochastic Model.", type=float, default='2.5', val='2.5'),
    Arg('T_FINAL', help="Final Temperature of the Stochastic Model.", type=float, default='0.01', val='0.01'),
    Arg('METHOD', help="Stochastic modelling method. It can be Metropolis or Simulated Annealing.", type=str, default='Annealing', val='Annealing'),
    Arg('FOLDING_COEFF', help="Folding coefficient.", type=float, default='1.0', val='1.0'),
    Arg('CROSS_COEFF', help="LEF crossing coefficient.", type=float, default='1.0', val='1.0'),
    Arg('BIND_COEFF', help="CTCF binding coefficient.", type=float, default='1.0', val='1.0'),

    # Molecular Dynamic Properties
    Arg('INITIAL_STRUCTURE_TYPE', help="you can choose between: rw, confined_rw, self_avoiding_rw, helix, circle, spiral, sphere.", type=str, default='rw', val='rw'),
    Arg('SIMULATION_TYPE', help="It can be either EM (multiple energy minimizations) or MD (one energy minimization and then run molecular dynamics).", type=str, default='EM', val='EM'),
    Arg('FORCEFIELD_PATH', help="Path to XML file with forcefield.", type=str, default='forcefield/classic_sm_ff.xml', val='forcefield/classic_sm_ff.xml'),
    Arg('ANGLE_FF_STRENGTH', help="Angle force strength.", type=float, default='200.0', val='200.0'),
    Arg('LE_FF_LENGTH', help="Equillibrium distance of loop forces.", type=float, default='0.0', val='0.0'),
    Arg('LE_FF_STRENGTH', help="Equillibrium distance of loop forces.", type=float, default='300000.0', val='300000.0'),
    Arg('EV_FF_STRENGTH', help="Excluded-volume strength.", type=float, default='10.0', val='10.0'),
    Arg('TOLERANCE', help="Tolerance that works as stopping condition for energy minimization.", type=float, default='0.001', val='0.001'),
    Arg('SIM_STEP', help="This is the amount of simulation steps that are perform each time that we change the loop forces. If this number is too high, the simulation is slow, if is too low it may not have enough time to adapt the structure to the new constraints.", type=int, default='100', val='100'),
])