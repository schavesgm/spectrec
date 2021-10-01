# Load some built-in modules
import os, re, json, datetime

# Load some third-party modules
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Load the basic classic
from spectrec.factory import SpectralDataset
from spectrec.losses  import MSELoss

# Load some utility functions and classes
from spectrec.utils   import SpectrecInput
from spectrec.utils   import train_network

# Peaks and kernels to be registered in the factory
from spectrec.network import UNet
from spectrec.factory import GaussianPeak
from spectrec.factory import DeltaPeak
from spectrec.factory import NRQCDKernel

# Load the factory functions to register some peaks and kernels
from spectrec.utils import register_peak_class
from spectrec.utils import register_kernel_class
from spectrec.utils import register_network_class

if __name__ == '__main__':

    # Register some peaks 
    register_peak_class('GaussianPeak', GaussianPeak)
    register_peak_class('DeltaPeak',    DeltaPeak)

    # Register some kernels
    register_kernel_class('NRQCDKernel', NRQCDKernel)

    # Register some networks
    register_network_class('UNet', UNet)

    # Load a dictionary of input parameters
    spectrec_input = SpectrecInput('./template.yml')

    # Save the information into tensorboard
    spectrec_input.write_to_tensorboard('./status/runs')

    # Device where the data will be stored
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get a dataset with the input parameters
    dataset = spectrec_input.get_dataset(verbose=True)

    # Get the network to be used in the training
    network = spectrec_input.get_network()

    # Load the parameters of the network if possible
    network.load_params('./status/network')

    # Generate the loss function used in the training
    loss = MSELoss()

    # Train the network using the parameters passed
    train_network(network, dataset, loss, spectrec_input.parse_train_info(), device, spectrec_input.run_name)

    # Save the network parameters
    network.save_params('./status/network')

    # Get the prefix and suffix of the dataset to use it in the saving
    dataset_info = spectrec_input.parse_dataset_info()
    prefix, suffix = dataset_info['prefix'], dataset_info['suffix']

    # Save the dataset
    dataset.save_dataset(prefix, suffix, './status/dataset')

    # # Save the network parameters to be used in the future
    # net.save_params(net_name, path=os.path.join(input_info['output'], 'network'))

    # # Validate the dataset
    # dataset.test(
    #     net, loss, network_info['val_prop'], device, network_info['batch_size'],
    #     'val', prefix=dataset_pre, suffix=dataset_suf, path=monitor_path
    # )

