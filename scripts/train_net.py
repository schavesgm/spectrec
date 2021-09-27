# Load some built-in modules
import os, re

# Load some third-party modules
import torch
import matplotlib.pyplot as plt

# Load some user-defined modules
from spectrec.factory import SpectralDataset
from spectrec.network import UNet
from spectrec.losses  import MSELoss
from spectrec.utils   import InputParser
from spectrec.utils   import train_network

if __name__ == '__main__':

    # Load a dictionary of input parameters
    input = InputParser('./input.yml').parse_input()

    # Get some needed properties of the input
    Nb, Ns, Nw, Nt = input['Nb'], input['Ns'], input['Nw'], input['Nt']

    # Device where the data will be stored
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate a kernel
    kernel = input['kernel'](Nt, Nw, input['w_range'])

    # Generate a dataset using the input dictionary
    dataset = SpectralDataset(
        input['peak_types'], input['peak_limits'], kernel,
        input['max_np'], input['fixed_np'], input['peak_ids']
    )

    # Get the prefix and suffix to be added to the dataset
    prefix, suffix = input['prefix'], input['suffix']

    # Generate the dataset name to check if it already exists
    name_id = dataset.name.replace('bNone_sNone', f'b{Nb}_s{Ns}')
    name_id = f'{prefix}_{name_id}' if prefix else name_id
    name_id = f'{name_id}_{suffix}' if suffix else name_id

    # Generate the dataset path using the location and the name identifier
    dataset_path = os.path.join(input['dataset_loc'], name_id)

    # Basis functions that will be used if basis file is passed
    basis = torch.load(input['basis_file']) if input['basis_file'] else None

    # Check if the given dataset already exists
    if os.path.exists(dataset_path):
        print(f' -- Dataset {name_id} exists')
        if input['overwrite']:
            if input['basis_file']:
                print(f'    -- Overwriting dataset using basis: ' + input['basis_file'])
            else:
                print(f'    -- Overwriting dataset')
            dataset.generate(Nb, Ns, basis=basis, use_GPU=input['use_GPU'])
        else:
            print(f'    -- Loading dataset')
            dataset.load_dataset(Nb, Ns, prefix, suffix, input['dataset_loc'])
    else:
        print(f' -- Generating {name_id} dataset')
        dataset.generate(Nb, Ns, basis=basis, use_GPU=input['use_GPU'])

    # Generate a network to be used in the training
    net = UNet(Nt, Ns)

    # Generate the network name using id + Ns + Nb + Nt + Nw + max_np + fixed_np
    net_name = input['net_id'] + '_' + re.sub('^b\d+_', '', dataset.name)

    # Load the network parameters
    net.load_params(net_name, input['network_loc'])

    # Generate a loss function to be used in the training
    loss = MSELoss()

    # Train the network
    train_network(net, dataset, loss, input['epochs'], device, input['batch_size'])

    # Save the network parameters to be used in the future
    net.save_params(net_name)

    # Test a portion of the dataset using the network
    dataset.test(net, loss, input['validation_prop'], device, prefix=prefix, suffix=suffix, batch_size=input['batch_size'])

    # Save the current dataset
    dataset.save_dataset(prefix, suffix, input['dataset_loc'])
