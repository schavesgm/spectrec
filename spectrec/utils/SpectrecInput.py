# Import built-in modules
import os, re, yaml
from collections import namedtuple
from datetime import date

# Import some third-party modules
import torch
from torch.utils.tensorboard import SummaryWriter

# Add some needed typing classes
from spectrec.factory import SpectralDataset
from spectrec.factory import Kernel
from spectrec.network import Network

# Factory functions to instantiate kernel and peak classes
from .NPKFactory import retrieve_kernel_class
from .NPKFactory import retrieve_peak_class
from .NPKFactory import retrieve_network_class

def validate_dataset(content: dict):
    """ Validate and spectrec's dataset input. """

    # Assert several keys are present in the dictionary
    assert all(k in content['dataset'] for k in ['id', 'parameters', 'generation', 'peaks', 'kernel'])

    # Get all subdictionaries from the data
    id_cont = content['dataset']['id']
    pa_cont = content['dataset']['parameters']
    ge_cont = content['dataset']['generation']
    pe_cont = content['dataset']['peaks']

    # Assert conditions on the id input
    assert all(k in id_cont for k in ['prefix', 'suffix'])
    assert isinstance(id_cont['prefix'], str)
    assert isinstance(id_cont['suffix'], str)

    # Assert conditions on the parameters input
    assert all(k in pa_cont for k in ['Nb', 'Nt', 'Ns', 'Nw', 'wr', 'mp', 'fp'])
    assert isinstance(pa_cont['Nb'], int) and pa_cont['Nb'] > 0
    assert isinstance(pa_cont['Nt'], int) and pa_cont['Nt'] > 0
    assert isinstance(pa_cont['Ns'], int) and pa_cont['Ns'] > 0
    assert isinstance(pa_cont['Nw'], int) and pa_cont['Nw'] > 0
    assert isinstance(pa_cont['mp'], int) and pa_cont['mp'] > 0
    assert isinstance(pa_cont['wr'], list)
    assert isinstance(pa_cont['fp'], bool)
    assert pa_cont['Nb'] > pa_cont['Nw']
    assert len(pa_cont['wr']) == 2
    assert all(isinstance(w, (float, int)) for w in pa_cont['wr'])

    # Assert conditions on the generation input
    assert all(k in ge_cont for k in ['overwrite', 'basis', 'use_GPU'])
    assert isinstance(ge_cont['overwrite'], bool)
    assert isinstance(ge_cont['use_GPU'], bool)
    assert isinstance(ge_cont['basis'], str)

    # Assert some conditions on the peaks used in the dataset
    assert all(k in pe_cont for k in ['peaks_used', 'limits'])
    assert isinstance(pe_cont['peaks_used'], list)
    assert isinstance(pe_cont['limits'], dict)
    assert all(isinstance(p, str) for p in pe_cont['peaks_used'])
    assert all(isinstance(v, list) for v in pe_cont['limits'].values())

    for val in pe_cont['limits'].values():
        assert len(val) == 2
        assert all(isinstance(p, (int, float)) for p in val)

    # Assert some conditions on the kernels used
    assert isinstance(content['dataset']['kernel'], str)

def validate_network(content: dict):
    """ Validate the network input. """
    assert (k in content['network'] for k in ['type', 'name'])
    assert isinstance(content['network']['type'], str)
    assert isinstance(content['network']['name'], str)

def validate_train(content: dict):
    """ Validate the train input. """
    # Get the train dictionary
    train = content['train']

    # Assert some keys are present
    assert (k in train for k in ['val_Nb', 'epochs', 'batch_size', 'lr_decay', 'num_valid'])

    # Assert some conditions on its contents
    assert isinstance(train['val_Nb'], int)     and train['val_Nb'] > 0
    assert isinstance(train['num_valid'], int)  and train['num_valid'] > 0
    assert isinstance(train['epochs'], int)     and train['epochs'] > 0
    assert isinstance(train['batch_size'], int) and train['batch_size'] > 0
    assert isinstance(train['lr_decay'], float)

class SpectrecInput:
    """ Class to parse an input file to obtain all needed input parameters. """

    def __init__(self, input_path: str):
        # Assert the data exists and load its contents
        assert os.path.exists(input_path), f'{input_path = } does not exist'
        with open(input_path, 'r') as input_file:
            self.__content = yaml.safe_load(input_file)

        # Validate all the input before continuing
        validate_dataset(self.__content)
        validate_network(self.__content)
        validate_train(self.__content)

        # Save the input file used to load the contents
        self.__input_file = input_path

    # -- Method to write some information to tensorboard
    def write_to_tensorboard(self, log_dir: str = './run'):
        """ Write all the information inside tensorboard. The localiser is the
        name of the run, that is, self.run_name.
        """

        # Open a writer and save all the needed information
        with SummaryWriter(os.path.join(log_dir, self.run_name)) as writer:

            # Save the name of the run, just to have it saved
            writer.add_text("run.name", self.run_name)

            # Save the dataset information
            for key, value in self.parse_dataset_info().items():
                writer.add_text("dataset." + key, str(value))

            # Save the network information
            writer.add_text("network.type", self.parse_network_info()['type'])
            writer.add_text("network.name", self.parse_network_info()['name'])

            # Save the training information
            for key, value in self.parse_train_info().items():
                writer.add_text("train." + key, str(value))

    # -- Parse methods {{{
    def parse_dataset_info(self) -> dict:
        """ Parse all the dataset information from the input file. """

        # Get all dataset information from the file
        dataset = self.__content['dataset']

        # Obtain the peak types and peak ids from the data
        peak_types, peak_ids = [], []

        for peak in dataset['peaks']['peaks_used']:

            # Get the peak name and its id
            matches = re.match(r'(\w+):?(.*)', peak)

            # Get the peak class to be used
            peak_types.append(matches.group(1))
            peak_ids.append(matches.group(2))

        # Get None instead of '' in the peak_ids
        peak_ids = list(map(lambda s: None if s == '' else s, peak_ids))

        # Return the dataset information as a dictionary
        return {
            'prefix':      dataset['id']['prefix'],
            'suffix':      dataset['id']['suffix'],
            'Nb':          self.dataset_params.Nb,
            'Nt':          self.dataset_params.Nt,
            'Nw':          self.dataset_params.Nw,
            'Ns':          self.dataset_params.Ns,
            'wr':          self.dataset_params.wr,
            'mp':          self.dataset_params.mp,
            'fp':          self.dataset_params.fp,
            'overwrite':   dataset['generation']['overwrite'],
            'basis':       dataset['generation']['basis'],
            'use_GPU':     dataset['generation']['use_GPU'],
            'peak_types':  peak_types,
            'peak_ids':    peak_ids,
            'peak_limits': dataset['peaks']['limits'],
            'kernel':      dataset['kernel']
        }

    def parse_network_info(self) -> dict:
        """ Parse all the network information from the input file. """
        return {
            'type': self.__content['network']['type'],
            'name': self.__content['network']['name']
        }

    def parse_train_info(self) -> dict:
        """ Parse all the train and validation information from input file. """
        return {
            'val_Nb':     self.__content['train']['val_Nb'],
            'num_valid':  self.__content['train']['num_valid'],
            'epochs':     self.__content['train']['epochs'],
            'batch_size': self.__content['train']['batch_size'],
            'lr_decay':   self.__content['train']['lr_decay'],
        }
    # -- }}}

    # -- Object initialiser methods {{{
    def get_kernel(self) -> Kernel:
        """ Generate a kernel object to be used in the run. """

        # Get the dataset information
        dataset = self.parse_dataset_info()

        # Get the properties of the class
        kernel  = retrieve_kernel_class(dataset['kernel'])

        return kernel(self.dataset_params.Nt, self.dataset_params.Nw, self.dataset_params.wr)

    def get_dataset(self, verbose: bool = False) -> SpectralDataset:
        """ Generate the already loaded/created dataset using the input parameters. """

        # Get the dataset information
        dataset_info = self.parse_dataset_info()

        # Get the parameters defining the SpectralDataset
        peak_types  = dataset_info['peak_types']
        peak_limits = dataset_info['peak_limits']
        mp          = dataset_info['mp']
        fp          = dataset_info['fp']
        peak_ids    = dataset_info['peak_ids']

        # Transform the peak_types to a list of registered Peaks
        peak_types = [retrieve_peak_class(p) for p in peak_types]

        # Generate the spectral dataset object to be loaded
        dataset = SpectralDataset(peak_types, peak_limits, self.get_kernel(), mp, fp, peak_ids)

        # Get the number of examples and coefficients in the dataset
        Nb, Ns = self.dataset_params.Nb, self.dataset_params.Ns

        # Get the prefix and suffix of the dataset
        prefix, suffix = dataset_info['prefix'], dataset_info['suffix']

        # Generate the dataset name to check if it already exists
        name = dataset.name.replace('bNone_sNone', f'b{Nb}_s{Ns}')
        name = f'{prefix}_{name}' if prefix else name
        name = f'{name}_{suffix}' if suffix else name

        # Generate the dataset path using the location and the name identifier
        dataset_path = os.path.join('status/dataset', name)

        # Load the basis functions if the basis functions file path is passed
        basis = torch.load(dataset_info['basis']) if dataset_info['basis'] else None

        # Function used to avoid printing indented lines
        def tell_me(listening: bool, comment: str):
            print(comment, flush=True) if listening else None

        # Check if the given dataset already exists
        if os.path.exists(dataset_path):
            tell_me(verbose, f' -- Dataset {name} exists')
            if dataset_info['overwrite']:
                if dataset_info['basis']:
                    tell_me(verbose, '    -- Overwriting dataset using basis: ' + dataset_info['basis'])
                else:
                    tell_me(verbose, '    -- Overwriting dataset')
                    dataset.generate(Nb, Ns, basis=basis, use_GPU=dataset_info['use_GPU'])
            else:
                tell_me(verbose, '    -- Loading dataset')
                dataset.load_dataset(Nb, Ns, prefix, suffix, './status/dataset')
        else:
            tell_me(verbose, f' -- Generating {name} dataset')
            dataset.generate(Nb, Ns, basis=basis, use_GPU=dataset_info['use_GPU'])

        return dataset

    def get_network(self) -> Network:
        """ Generate a network object using the parameters in the run. """
        
        # Get the network information
        network_info = self.parse_network_info()
        
        # Get the network type
        network = retrieve_network_class(network_info['type'])

        return network(self.dataset_params.Nt, self.dataset_params.Ns, name=network_info['name'])
    # -- }}}

    # -- Property methods of the class {{{
    @property
    def dataset_params(self) -> namedtuple:
        """ Return the basic parameters of the run. """

        # Parameters named tuple to be returned
        Parameters = namedtuple('Parameters', 'Nb, Ns, Nt, Nw, wr, mp, fp')

        # Obtain the parameters part of the dictionary
        params = self.__content['dataset']['parameters']

        # Obtain the parameters from the dictionary
        return Parameters(
            params['Nb'], params['Ns'], params['Nt'], params['Nw'], 
            sorted(params['wr']), params['mp'], params['fp']
        )

    @property
    def run_name(self) -> str:
        """ Generate the run name identifier using the network and dataset. """

        # Get all parameters and network information
        params, network = self.dataset_params, self.parse_network_info()

        # Get the id part of the dataset
        prefix = self.__content['dataset']['id']['prefix']
        suffix = self.__content['dataset']['id']['suffix']

        # Generate the generate name of the simulation
        name = f'b{params.Nb}_s{params.Ns}_t{params.Nt}'
        name = f'{name}_w{params.Nw}_mp{params.mp}_fp{int(params.fp)}'
        name = f'{name}_wi{params.wr[0]:.2f}_wf{params.wr[1]:.2f}'

        # Add the prefix and suffix if needed
        name = f'{prefix}_{name}' if prefix else name
        name = f'{name}_{suffix}' if suffix else name

        return network['type'] + '/' + network['name'] + '/' + str(date.today()) + '_' + name
    # -- }}}

if __name__ == '__main__':
    pass
