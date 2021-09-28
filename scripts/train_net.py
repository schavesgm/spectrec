# Load some built-in modules
import os, re

# Load some third-party modules
import torch
import matplotlib.pyplot as plt

# Load the basic classic
from spectrec.factory import SpectralDataset
from spectrec.network import UNet
from spectrec.losses  import MSELoss

# Load some utility functions and classes
from spectrec.utils   import InputParser
from spectrec.utils   import train_network

# Peaks and kernels to be registered in the factory
from spectrec.factory import GaussianPeak
from spectrec.factory import DeltaPeak
from spectrec.factory import NRQCDKernel

# Load the factory functions to register some peaks and kernels
from spectrec.utils import register_peak_class
from spectrec.utils import register_kernel_class

if __name__ == '__main__':

    # Register some peaks 
    register_peak_class('GaussianPeak', GaussianPeak)
    register_peak_class('DeltaPeak',    DeltaPeak)

    # Register some kernels
    register_kernel_class('NRQCDKernel', NRQCDKernel)

    # Load a dictionary of input parameters
    input_info = InputParser('./input.yml').parse_input()

    # Get the information related to the dataset and network
    dataset_info, network_info = input_info['dataset'], input_info['network']

    # Get some important parameters from the dataset information
    Nb, Ns = dataset_info['Nb'], dataset_info['Ns']
    Nw, Nt = dataset_info['Nw'], dataset_info['Nt']

    # Device where the data will be stored
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate a kernel
    kernel = dataset_info['kernel'](Nt, Nw, dataset_info['w_range'])

    # Generate a dataset using the input dictionary
    dataset = SpectralDataset(
        dataset_info['peak_types'], dataset_info['peak_limits'], kernel, 
        dataset_info['max_np'], dataset_info['fixed_np'], dataset_info['peak_ids']
    )

    # Get the prefix and suffix to be added to the dataset
    dataset_pre, dataset_suf = dataset_info['prefix'], dataset_info['suffix']

    # Generate the dataset name to check if it already exists
    name_id = dataset.name.replace('bNone_sNone', f'b{Nb}_s{Ns}')
    name_id = f'{dataset_pre}_{name_id}' if dataset_pre else name_id
    name_id = f'{name_id}_{dataset_suf}' if dataset_suf else name_id

    # Generate the dataset path using the location and the name identifier
    dataset_path = os.path.join(input_info['output'], 'dataset', name_id)

    # Load the basis functions if the basis functions file path is passed
    basis = torch.load(dataset_info['basis']) if dataset_info['basis'] else None

    # Check if the given dataset already exists
    if os.path.exists(dataset_path):
        print(f' -- Dataset {name_id} exists')
        if dataset_info['overwrite']:
            if dataset_info['basis']:
                print(f'    -- Overwriting dataset using basis: ' + dataset_info['basis'])
            else:
                print(f'    -- Overwriting dataset')
            dataset.generate(Nb, Ns, basis=basis, use_GPU=dataset_info['use_GPU'])
        else:
            print(f'    -- Loading dataset')
            dataset.load_dataset(Nb, Ns, dataset_pre, dataset_suf, os.path.join(input_info['output'], 'dataset'))
    else:
        print(f' -- Generating {name_id} dataset')
        dataset.generate(Nb, Ns, basis=basis, use_GPU=dataset_info['use_GPU'])

    # Generate a network to be used in the training
    net = UNet(Nt, Ns)

    # Generate the network name using id + Ns + Nb + Nt + Nw + max_np + fixed_np
    net_name = re.sub(r'^b\d+_', '', dataset.name)
    net_name = network_info['prefix'] + '_' + net_name if network_info['prefix'] else net_name

    # Load the network parameters
    net.load_params(net_name, path=os.path.join(input_info['output'], 'network'))

    # Generate a loss function to be used in the training
    loss = MSELoss()

    # Monitor path
    monitor_path = os.path.join(input_info['output'], 'monitor', net.net_id, net_name)

    # Train the network
    train_network(
        net, dataset, loss, network_info['epochs'], device, network_info['batch_size'], path=monitor_path
    )

    # Save the network parameters to be used in the future
    net.save_params(net_name, path=os.path.join(input_info['output'], 'network'))

    # Validate the dataset
    dataset.test(
        net, loss, network_info['val_prop'], device, network_info['batch_size'],
        'val', prefix=dataset_pre, suffix=dataset_suf, path=monitor_path
    )

    # Save the current dataset
    dataset.save_dataset(dataset_pre, dataset_suf, path=os.path.join(input_info['output'], 'dataset'))
