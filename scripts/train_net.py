# Load some built-in modules
import argparse

# Load some third-party modules
import torch

# Load some needed spectrec utilities
from spectrec.utils   import SpectrecInput
from spectrec.losses  import MSELoss

# Load the distributed architecture setup functions
from spectrec.training import initialise_dist_nodes
from spectrec.training import initiliase_dist_gpu
from spectrec.training import Trainer
from spectrec.training import generate_validation_sets

def train(gpu: int, config_file: str, node_info: dict):
    """ Main train function to be spawned in the distributed architecture. """

    # Initialise the distributed gpu
    rank = initiliase_dist_gpu(gpu, node_info)

    # Generate a spectrec input loader
    spectrec_input = SpectrecInput(config_file)

    # Register the external classes in the system
    spectrec_input.register_classes(verbose=True)

    # Get the training info from the loader
    train_info   = spectrec_input.parse_train_info()
    dataset_info = spectrec_input.parse_dataset_info()

    # Save the information into tensorboard
    spectrec_input.write_to_tensorboard('./status/runs')

    # Get a dataset with the input parameters
    dataset = spectrec_input.get_dataset(verbose=True)

    # Save the dataset if possible
    dataset.save_dataset(dataset_info['prefix'], dataset_info['suffix'])

    # Generate a sampler to be used in the training
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, num_replicas=node_info['world_size'], rank=rank, seed=31
    )

    # Generate a loader to be used in the training
    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=train_info['batch_size'], 
        num_workers=node_info['workers'], pin_memory=True, drop_last=True
    )
    
    # Generate several validation sets
    valid_sets = generate_validation_sets(dataset, train_info, node_info, rank)

    # Generate the model to be used in the training
    network = spectrec_input.get_network().cuda(gpu)

    # Synchronise the batch normalisation
    network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)

    # Wrap the model around a distributed dataparallel
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[gpu])

    # Generate the loss function. 
    loss = MSELoss()

    # Generate the optimiser to be used in the training. 
    optim = torch.optim.Adam(network.parameters())

    # Generate a training object to train the network
    Trainer(train_info, loader, network, loss, optim, (rank == 0)).train_and_validate(valid_sets)

if __name__ == '__main__':

    # Decription message of the program
    desc_msg = 'Train script to train a network for spectral reconstruction. The training can be distributed' + \
               'among different nodes and GPUs to allow more efficiency.'

    # Use a command line argument parser
    parser = argparse.ArgumentParser(description=desc_msg)

    # Add some arguments to the parser
    parser.add_argument('-config', type=str, default='./template.yml', help='Path to configuration yaml file')
    parser.add_argument('-gpus', type=str, default='0', help='Comma separated gpu device indices.')
    parser.add_argument('-nodes', type=int, default=1, help='Number of nodes used in the distributed architecture.')
    parser.add_argument('-node_rank', type=int, default=0, help='Rank corresponding to the current node.')
    parser.add_argument('-workers', type=int, default=0, help='Number of subprocesses used when loading data.')

    # Parse the arguments
    args = parser.parse_args()

    # Initialise the distributed nodes
    node_info = initialise_dist_nodes(args.gpus, args.nodes, args.node_rank, args.workers)

    # Spawn some distributed training.
    torch.multiprocessing.spawn(train, args=(args.config, node_info), nprocs=node_info['gpus_per_node'])
