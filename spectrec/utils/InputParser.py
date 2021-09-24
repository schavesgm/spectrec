# Import built-in modules
import os, re, yaml

# Import user defined modules
from spectrec.factory import Peak
from spectrec.factory import Kernel
from spectrec.factory import GaussianPeak
from spectrec.factory import DeltaPeak
from spectrec.factory import NRQCDKernel

class InputParser:
    """ Class to parse a file to obtain an input. """

    # Registered peaks to be used in the class
    __reg_peaks = [GaussianPeak, DeltaPeak]

    # Registered kernels that can be used in the class
    __reg_kernels = [NRQCDKernel]

    def __init__(self, path_to_input_file: str):

        # Check the file exists
        assert os.path.exists(path_to_input_file), f'{path_to_input_file = } does not exist'

        # Load the yaml file
        with open(path_to_input_file, 'r') as input_file:
            self.__content = yaml.safe_load(input_file)

    def parse_input(self) -> dict:
        """ Parse all information in the dataset. """
        return self.parse_dataset() | self.parse_network()

    def parse_dataset(self) -> dict:
        """ Parse the dataset information from the file. """

        # Assert dataset exists in the input keys
        assert 'dataset' in self.__content.keys(), 'Dataset is not present in input.'''

        # Dataset entry in the dictionary
        dataset = self.__content['dataset']

        # Get the peaks used in the dataset generation
        peak_used = dataset['peaks']['peak_used']

        # List that will contain all used peaks in the simulation
        peak_types = []
        peak_ids   = []

        # Iterate for each peak in the dictionary and its name
        for peak in peak_used:

            # Match the peak type and the name of the peak
            matches = re.match(r'(\w+):?(.*)', peak)

            # Get the peak that will be used in the input
            peak_type = self.__class_from_str(matches.group(1), self.__reg_peaks)

            # Get the peak identifier
            peak_id = matches.group(2)

            # Append a new peak to the input
            peak_types.append(peak_type)
            peak_ids.append(peak_id)

        # Transform all '' to None
        peak_ids = list(map(lambda s: None if s == '' else s, peak_ids))

        # Return the dataset information as a dictionary
        return {
            'prefix':      dataset['id']['prefix'],
            'suffix':      dataset['id']['suffix'],
            'dataset_loc': dataset['id']['location'],
            'overwrite':   dataset['id']['overwrite'],
            'basis_file':  dataset['basis']['file'],
            'use_GPU':     dataset['basis']['use_GPU'],
            'Nb':          dataset['basic']['Nb'],
            'Nt':          dataset['basic']['Nt'],
            'Nw':          dataset['basic']['Nw'],
            'Ns':          dataset['basic']['Ns'],
            'w_range':     dataset['basic']['w_range'],
            'max_np':      dataset['peaks']['max_npeaks'],
            'fixed_np':    dataset['peaks']['fixed_npeaks'],
            'peak_limits': dataset['peaks']['limits'],
            'peak_types':  peak_types,
            'peak_ids':    peak_ids,
            'kernel':      self.__class_from_str(dataset['kernel'], self.__reg_kernels)
        }

    def parse_network(self) -> dict:
        """ Parse all information related to the network. """

        # Assert network exists in the YAML file
        assert 'network' in self.__content.keys(), 'Network is not present in input.'''

        # Get the network information
        network = self.__content['network']

        return {
            'net_id':          network['id'],
            'validation_prop': network['validation_prop'],
            'epochs':          network['epochs'],
            'batch_size':      network['batch_size'],
            'network_loc':     network['location'],
        }

    def register_peaks(self, *peaks: Peak) -> None:
        """ Register some new peak classes to be parsed. """
        for peak in peaks:
            self.__reg_peaks.append(peak)

    def register_kernel(self, *kernels: Kernel) -> None:
        """ Register some new kernel classes to be parsed. """
        for kernel in kernels:
            self.__reg_kernels.append(kernel)

    # -- Private methods of the class {{{
    def __class_from_str(self, class_str: str, reg_obj: list[object]) -> object:
        """ Get a class using the string definition. """
        # Index that matches peak_type in peak_list
        for def_class in reg_obj:
            if class_str in str(def_class):
                return def_class
    # -- }}}

    @property
    def registered_peaks(self) -> list[Peak]:
        ''' Show all registered peaks in the class. '''
        return self.__reg_peaks

    @property
    def registered_kernels(self) -> list[Kernel]:
        ''' Show all registered kernels in the class. '''
        return self.__reg_kernels

if __name__ == '__main__':
    pass
