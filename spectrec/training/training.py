# Import some built-in modules
import os
import math
import sys
import argparse

from glob import glob

# Import some third-party modules
import torch

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data             import DataLoader
from torch.nn                     import Module
from torch.optim                  import Optimizer
from torch.utils.tensorboard      import SummaryWriter
from torch.optim.lr_scheduler     import _LRScheduler

# Load some spectrec modules
from spectrec.factory import SpectralDataset

# Load some local modules
from .loggers import TrainLogger
from .plots   import plot_example

class Trainer:
    """ Trainer class used to train a network on a distributed architecture """

    def __init__(self, args: argparse.Namespace, train_loader: DataLoader, model: Module, loss: Module, optimiser: Optimizer):

        # Save all the argument information in the module
        self.args = args
        
        # Save some important information in the class
        self.model        = model
        self.optimiser    = optimiser
        self.loss         = loss
        self.train_loader = train_loader

        # Set the fp16 scaler if contained inside train information
        self.fp16_scaler = torch.cuda.amp.GradScaler() if self.args.fp16 else None

        # If this is the main node, then generate some writers
        if self.args.is_main:
            self.writer = SummaryWriter(os.path.join('./status/runs/{}'.format(self.args.run_name)))

    def train_one_epoch(self, epoch: int, lr_schedule: _LRScheduler):
        """ Train one epoch using a lr_scheduler. """

        # Generate a logger to output some data
        logger = TrainLogger(log_every=self.args.log_every)

        # Iterate for each example in the training loader
        for it, (input_data, label_data) in enumerate(logger.log_information(self.train_loader, header=f'Epoch :{epoch}')):

            # Move the inputs to cuda
            input_data = input_data.cuda(non_blocking=True).log()
            label_data = label_data.cuda(non_blocking=True)

            # Forward pass of the network using fp16 if needed
            with torch.cuda.amp.autocast(self.args.fp16):
                preds = self.model(input_data)
                loss  = self.loss(preds, label_data, input_data)

            # Sanity check
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training.".format(loss.item()), flush=True)
                sys.exit(1)

            # Backward pass of the model
            self.model.zero_grad()

            # If we are using fp16, then scale the data
            if self.args.fp16:
                self.fp16_scaler.scale(loss).backward()
                self.fp16_scaler.step(self.optimiser)
                self.fp16_scaler.update()
            else:
                loss.backward()
                self.optimiser.step()

            # Log some data to the console
            torch.cuda.synchronize()

            # Update the logger with the new loss function value
            logger.update_loss(loss.item())

            # Delete the tensors
            del input_data, label_data, preds

            # Log some data to tensorboard
            if self.args.is_main:
                self.writer.add_scalar("train/loss", loss.item(), epoch * len(self.train_loader) + it)

        # Synchronise the logger between processes
        logger.synchronise_between_processes()

    def train_and_validate(self, validate_loaders: list[DataLoader]):
        """ Train the model using the data provided and validate it using the validate sets
        contained inside validate_loaders.
        """

        # Generate several validate writers to log some information to tensorboard
        if self.args.is_main:
            valid_writers = [
                SummaryWriter('./status/runs/{}/validate_{}'.format(self.args.run_name, v)) \
                for v in range(len(validate_loaders))
            ]

        # Load the data to resume a training if possible
        self.load_if_possible()

        # Generate a lr scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=10)

        # Training loop
        for epoch in range(self.start_epoch, self.args.epochs):

            # Generate the dataset
            self.train_loader.sampler.set_epoch(epoch)

            # Train this epoch
            self.train_one_epoch(epoch, lr_scheduler)

            # Validation log format
            valid_log = '  ** Valid ({epoch}, {valid}): loss: {loss:.4f}, dL: {dL:.4f}, dR: {dR:.4f}'

            # Iterate for each validation loader to validate the training
            for nv, valid in enumerate(validate_loaders):

                # Obtain the dataset to be validated
                valid.sampler.set_epoch(epoch)

                # Get the total loss, total dL and dR
                total_loss, total_dL, total_dR = 0.0, 0.0, 0.0

                for it, (input_data, label_data) in enumerate(valid):

                    # Move the inputs to cuda
                    input_data = input_data.cuda(non_blocking=True).log()
                    label_data = label_data.cuda(non_blocking=True)

                    # Forward pass of the network
                    preds = self.model(input_data)
                    total_loss += self.loss(preds, label_data, input_data).item()

                    # Compute the MSError in the coefficients
                    total_dL += ((label_data - preds)[:, 0, :] ** 2).mean().item()

                    # Generate the difference in spectral functions
                    total_dR += ((label_data - preds)[:, 0, :] @ valid.dataset.U.T.cuda()).abs().max(axis=1).values.mean().item()

                    # Synchronise the CUDA devices
                    torch.cuda.synchronize()

                    # Plot some examples in the last iteration for the main
                    if (it + 1) == len(valid) and self.args.is_main:

                        # Monitor some examples in the validation set
                        for nf in range(3):

                            # Construct a figure to be saved
                            figure = plot_example(
                                label_data[nf, 0,:].detach().cpu(), 
                                preds[nf, 0, :].clone().detach().cpu(),
                                valid.dataset.U.T, valid.dataset.kernel.omega, epoch, nf
                            )

                            # Add the figure to the writer
                            self.writer.add_figure(f"train/validate{nv}_ex{nf}", figure, epoch)

                    # Delete some tensors
                    del input_data, label_data, preds

                # Log some information to the console
                print(
                    valid_log.format(
                        epoch=epoch, valid=nv, loss=total_loss/len(valid), 
                        dL=total_dL/len(valid), dR=total_dR/len(valid)
                    ), flush=True
                )

                # If this is the original rank, then write some stuff to tensorboard
                if self.args.is_main:
                    valid_writers[nv].add_scalar("train/loss", total_loss / len(valid), len(self.train_loader) * (epoch + 1))
                    valid_writers[nv].add_scalar("train/dL",   total_dL   / len(valid), epoch)
                    valid_writers[nv].add_scalar("train/dR",   total_dR   / len(valid), epoch)

            # Add the learning rate to tensorboard
            if self.args.is_main:
                self.writer.add_scalar("train/lr", torch.Tensor(lr_scheduler.get_last_lr()), len(self.train_loader) * (epoch + 1))

            # Step the lr_scheduler
            lr_scheduler.step()

            # Save the data every some epochs to be able to resume it
            if self.args.is_main and epoch % self.args.save_every == 0:
                self.save_information(epoch)

    def load_if_possible(self):
        """ Load information to resume training. """

        # Get the directories where the data is stored
        runs_dirs = sorted(glob('./status/weights/{}/Epoch_*.pth'.format(self.args.run_name)))

        if len(runs_dirs) == 0:
            self.start_epoch = 0
            print("Starting training from scratch")
        else:
            # Load the last epoch information
            info = torch.load(runs_dirs[-1], map_location='cpu')
            self.start_epoch = info['epoch']
            self.model.load_state_dict(info['model'])
            self.optimiser.load_state_dict(info['optimiser'])
            if self.args.fp16:
                self.fp16_scaler.load_state_dict(info['fp16_scaler'])
            print('Loaded information ', runs_dirs[-1])

    def save_information(self, epoch: int):
        """ Save the information of the network """

        # Dictionary containin all the data
        state = dict(
            epoch=epoch+1,
            model=self.model.state_dict(),
            optimiser=self.optimiser.state_dict(),
            args=self.args,
        )

        # Add the floating scaler if used
        if self.args.fp16:
            state['fp16_scaler'] = self.fp16_scaler.state_dict()

        # Get the output path
        out = './status/weights/{}/'.format(self.args.run_name)

        # Create the path if it does not exist
        if not os.path.exists(out): os.makedirs(out)

        # Save the dictionary
        torch.save(state, '{}/Epoch_{}.pth'.format(out, str(epoch).zfill(3)))

def generate_validation_sets(dataset: SpectralDataset, args) -> list[DataLoader]:

    # List that will contain the validation sets
    valid_list = []

    for nv in range(args.num_valid):

        # Generate a spectral dataset with the same configuration
        valid = SpectralDataset(
            dataset.peak_types, dataset.peak_limits, dataset.kernel,
            dataset.max_np, dataset.fixed_np, dataset.peak_ids
        )

        # Generate the data inside the spectral dataset
        valid.generate(args.val_Nb, dataset.Ns, dataset.U)

        # Sampler used in the distributed architecture
        sampler = DistributedSampler(
            valid, shuffle=False, num_replicas=args.world_size, rank=args.rank, seed=args.seed
        )

        # Append a dataloader around the validation set
        loader = DataLoader(
            valid, sampler=sampler, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )

        # Append the loader to the validation list
        valid_list.append(loader)

    return valid_list

if __name__ == '__main__':
    pass
