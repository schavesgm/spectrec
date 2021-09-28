# Import built-in modules
import os, re, yaml

# Factory functions to instantiate kernel and peak classes
from .PKFactory import create_kernel_class
from .PKFactory import create_peak_class

class InputParser:
    """ Class to parse an input file to obtain all needed input parameters. """

    def __init__(self, input_path: str):
        # Assert the data exists and load its contents
        assert os.path.exists(input_path), f'{input_path = } does not exist'
        with open(input_path, 'r') as input_file:
            self.__content = yaml.safe_load(input_file)

        # Save the input file used to load the contents
        self.__input_file = input_path

    def parse_input(self) -> dict:
        """ Parse the whole input information. """
        return {
            'input':   self.__input_file,
            'output':  self.__content['output_path'],
            'dataset': self.parse_dataset(),
            'network': self.parse_network(),
        }

    def parse_dataset(self) -> dict:
        """ Parse all dataset information from the file. """

        # Get all dataset information from the file
        dataset = self.__content['dataset']

        # Obtain the peak types and peak ids from the data
        peak_typ, peak_ids = [], []

        for peak in dataset['peaks']['peaks_used']:

            # Get the peak name and its id
            matches = re.match(r'(\w+):?(.*)', peak)

            # Get the peak class to be used
            peak_typ.append(create_peak_class(matches.group(1)))
            peak_ids.append(matches.group(2))

        # Get None instead of '' in the peak_ids
        peak_ids = list(map(lambda s: None if s == '' else s, peak_ids))

        # Return the dataset information as a dictionary
        return {
            'prefix':      dataset['id']['prefix'],
            'suffix':      dataset['id']['suffix'],
            'Nb':          dataset['parameters']['Nb'],
            'Nt':          dataset['parameters']['Nt'],
            'Nw':          dataset['parameters']['Nw'],
            'Ns':          dataset['parameters']['Ns'],
            'w_range':     dataset['parameters']['w_range'],
            'overwrite':   dataset['generation']['overwrite'],
            'basis':       dataset['generation']['basis'],
            'use_GPU':     dataset['generation']['use_GPU'],
            'peak_types':  peak_typ,
            'peak_ids':    peak_ids,
            'peak_limits': dataset['peaks']['limits'],
            'max_np':      dataset['peaks']['max_np'],
            'fixed_np':    dataset['peaks']['fixed_np'],
            'kernel':      create_kernel_class(dataset['kernel'])
        }

    def parse_network(self) -> dict:
        """ Parse all the network information from the input file. """

        # Get the network information
        network = self.__content['network']

        return {
            'prefix':      network['prefix'],
            'val_prop':    network['val_prop'],
            'epochs':      network['epochs'],
            'batch_size':  network['batch_size'],
        }

if __name__ == '__main__':
    pass
