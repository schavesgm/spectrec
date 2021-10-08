# -- Most code in this file is based on: git@github.com:ramyamounir/Template.git

# Import built-in modules
import os
import random
import re
import signal
import subprocess
import argparse
from pathlib import Path

# Import third-party modules
import torch
import submitit
import torch.distributed as dist
import torch.backends.cudnn as cudnn

# -- Utility functions {{{
def disable_print_for_non_main(is_main: bool, print_all: bool):
    """ Disable printing for non-master (rank=0) processes """
    import builtins as __builtins__
    builtin_print = __builtins__.print

    # If print_all is enabled, then do nothing
    if print_all: return

    # If not, remove the print for non-main processes
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)
    
    __builtins__.print = print

def fix_random_seeds(seed: int = 31):
    """ Fix the random seeds for torch random numbers. 

    --- Parameters:
    seed: int
        Seed to be used in the program.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_slurm_output_directory(shared_path: str) -> Path:
    """ Get the slurm output directory used by all nodes and GPUs """

    # Check if a directory exists, if not, just create it
    if not os.path.exists(shared_path): os.makedirs(shared_path)

    # The output will be in an specific USER folder
    p = Path(os.path.join(shared_path, os.getenv('USER')))

    # Create the user folder if it does not exist
    p.mkdir(exist_ok=True)

    return p
# -- }}}

# -- Function handlers to handle some signals {{{
def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass
# -- }}}

def initialise_dist_nodes(args: argparse.Namespace):
    """ Function used to initialise the distributed nodes. It acts differently
    depending on whether we are training locally on in a SLURM server. It
    sets the number of gpus per node, the url used to communicate between
    processes and the rank and world_size of each GPU.
    """

    # If the job is a SLURM job, then act differently
    if 'SLURM_JOB_ID' in os.environ:

        # Number of GPUS per node
        args.ngpus_per_node = torch.cuda.device_count()

        # Requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

        # Find a common host name on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]

        # Url used in the distribution
        args.url = f'tcp://{host_name}:{args.port}'

        # Distributed parameters
        args.rank       = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node

    else:

        # Set the all visible cuda devices
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

        # Define some important variables
        args.ngpus_per_node = torch.cuda.device_count()
        args.rank           = 0
        args.url            = f'tcp://localhost:{args.port}'
        args.world_size     = args.ngpus_per_node

def initiliase_dist_gpu(gpu: int, args: argparse.Namespace) -> int:
    """ Function to set the code for a distributed GPU. The functions acts differently
    depending on whether we are training on a SLURM server or locally. The function
    sets the gpu and rank for of the given GPU and also initialises the group process.
    """

    # Act differently depending on type of job
    if args.slurm:

        # Get the job environment
        job_env = submitit.JobEnvironment()

        # Update the output directory with the jobID
        args.slurm_shared_dir = Path(str(args.slurm_shared_dir).replace('%j', str(job_env.job_id)))

        # Get the local rank (GPU) and the global rank (node * world_size + GPU)
        args.gpu  = job_env.local_rank
        args.rank = job_env.global_rank
    else:
        
        # Get the local rank (GPU) and the global rank (GPU)
        args.gpu   = gpu
        args.rank  += gpu

    # Initialise the process group of this gpu
    dist.init_process_group(backend=args.backend, init_method=args.url, world_size=args.world_size, rank=args.rank)

    # Fix all random seeds for this device
    fix_random_seeds(args.seed)

    # Set the cuda devices
    torch.cuda.set_device(args.gpu)

    # Set some properties in the backend
    cudnn.benchmark = True

    # Synchronise all devices
    dist.barrier()

    # Set if the current process is the main one
    args.is_main = (args.rank == 0)

    # Disable printing for non master ranks
    disable_print_for_non_main(args.is_main, args.print_all)

if __name__ == '__main__':
    pass
