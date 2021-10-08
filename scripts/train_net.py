# Load some built-in modules
import argparse
import random
from pathlib import Path

# Load some third-party modules
import torch
import submitit

# Load some needed spectrec utilities
from spectrec.utils   import SpectrecInput
from spectrec.losses  import MSELoss

# Load the distributed architecture setup functions
from spectrec.training import initialise_dist_nodes
from spectrec.training import initiliase_dist_gpu
from spectrec.training import Trainer
from spectrec.training import generate_validation_sets
from spectrec.training import get_slurm_output_directory

def train(gpu: int, args):
    """ Main train function to be spawned in the distributed architecture. """

    # Initialise the distributed gpu
    initiliase_dist_gpu(gpu, args)

    # Generate a spectrec input loader
    spectrec_input = SpectrecInput(args.config)

    # Register the external classes in the system
    spectrec_input.register_classes(verbose=True)

    # Set some needed information from the input into args
    for key, value in spectrec_input.parse_train_info().items():
        setattr(args, key, value)

    for key, value in spectrec_input.parse_dataset_info().items():
        setattr(args, key, value)

    for key, value in spectrec_input.parse_network_info().items():
        setattr(args, key, value)

    # Save the information into tensorboard
    spectrec_input.write_to_tensorboard('./status/runs')

    # Get a dataset with the input parameters
    dataset = spectrec_input.get_dataset(verbose=True)

    # Save the dataset if possible
    dataset.save_dataset(args.prefix, args.suffix)

    # Generate a sampler to be used in the training
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, num_replicas=args.world_size, rank=args.rank, seed=args.seed
    )

    # Generate a loader to be used in the training
    loader = torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    
    # Generate the model to be used in the training
    network = spectrec_input.get_network().cuda(args.gpu)
    network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[args.gpu])

    # Generate the loss function.
    loss = MSELoss().cuda(args.gpu)

    # Generate the optimiser to be used in the training.
    optim = torch.optim.Adam(network.parameters())

    # Generate a training object to train the network
    Trainer(args, loader, network, loss, optim).train_and_validate(
        generate_validation_sets(dataset, args)
    )

if __name__ == '__main__':

    # Decription message of the program
    desc_msg = 'Training script for spectral reconstruction. It allows distributed data parallel jobs.' + \
        'There are two different distributed architectures available: local and SLURM. In local, one can' + \
        'train on several GPUs but only one node. In contrast, in SLURM, one can submit SLURM jobs. As a' + \
        'consequence, the training can be parallelised on several nodes each with a given number of GPUs.'

    # Use a command line argument parser
    parser = argparse.ArgumentParser(description=desc_msg)

    # Add some common arguments to the parser
    parser.add_argument('-config', type=str, default='./template.yml', help='Path to configuration yaml file')
    parser.add_argument('-seed', type=int, default=31, help='Seed to be used in the random engines')
    parser.add_argument('-backend', type=str, default='nccl', help='Used torch distributed backend.')
    parser.add_argument('-print_all', action='store_true', default=False, help='Log information with all processes.')
    parser.add_argument('-workers', type=int, default=2, help='Number of subprocesses used when loading data.')
    
    # Add some local run information
    parser.add_argument('-gpus', type=str, default='0', help='Comma separated gpu device indices. Only useful if -slurm is not passed.')

    # Add the slurm information to the parser
    parser.add_argument('-slurm', action='store_true', help='Use SLURM to train the model.')
    parser.add_argument('-slurm_ngpus', type=int, default=2, help='Number of GPUs per node in SLURM')
    parser.add_argument('-slurm_nnodes', type=int, default=1, help='Number of nodes used in SLURM')
    parser.add_argument('-slurm_partition', type=str, default='gpu', help='Partition where the SLURM job will be submitted')
    parser.add_argument('-slurm_shared_dir', type=str, help='Directory where SLURM output will be output')

    # Parse the arguments
    args = parser.parse_args()

    # Add some information to the arguments
    args.port = random.randint(49152, 65535)

    # If we are running on the cloud, then use the SLURM movement
    if args.slurm:
        # Generate a SLURM class to wrap the distributed execution
        class SLURMTrainer:
            def __init__(self, args):
                self.args = args

            def __call__(self):
                initialise_dist_nodes(self.args)
                train(None, self.args)

        # Generate the slurm output folder and attach the job identifier
        args.slurm_shared_dir = get_slurm_output_directory(args.slurm_shared_dir)

        # Append a localiser to change each jobID
        args.slurm_shared_dir = args.slurm_shared_dir / "%j"

        # Generate a submitit executor
        executor = submitit.AutoExecutor(folder=args.slurm_shared_dir, slurm_max_num_timeout=30)

        # Update some parameters on submitit
        executor.update_parameters(
            gpus_per_node=args.slurm_ngpus,
            tasks_per_node=args.slurm_ngpus,
            cpus_per_task=2,
            nodes=args.slurm_nnodes,
            timeout_min=2800,
            slurm_partition=args.slurm_partition,
            name="Spectrec",
        )

        # Execute the job
        job = executor.submit(SLURMTrainer(args))
        print(f'Submitted job_id: {job.job_id}')

    # If not, use the normal distributed dataparallel
    else:
        # Initialise the distributed nodes
        initialise_dist_nodes(args)

        # Spawn some distributed training.
        torch.multiprocessing.spawn(train, args=(args,), nprocs=args.ngpus_per_node)
