# Load some built-in modules
import os

# Load some third-party modules
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.legend_handler import HandlerTuple
from torch.utils.tensorboard import SummaryWriter

# Load some user-defined modules
from spectrec.factory import SpectralDataset
from spectrec.network import Network

# Load some utility modules
from .plots import eliminate_mirror_axis

# Color used in the example plots
COLOR_1, COLOR_2, COLOR_3 = '#1E6091', '#40916C', '#BA181B'

def plot_validate_example(
        Ll: torch.Tensor, Lp: torch.Tensor, Rl: torch.Tensor, Rp: torch.Tensor, omega: torch.Tensor, ne: int, ex: int = 0
    ) -> plt.Figure:
    """ Plot a random validation example to monitor the spectral reconstruction. The output
    contains a figure with three different axis: one comparing label and predicted coefficients,
    one comparing the absolute difference between labels and coefficients and the last one comparing
    the label and predicted spectral functions.

    --- Parameters:
    Ll: torch.Tensor
        Label coefficients.
    Lp: torch.Tensor
        Predicted coefficients.
    Rl: torch.Tensor
        Label spectral functions.
    Rp: torch.Tensor
        Predicted spectral functions.
    omega: torch.Tensor
        Omega values to make the spectral function plot prettier.
    ne: int
        Number to write on the label to make the plots different. Tensorboard hack.
    ex: int
        Example to be plotted from the data

    --- Returns:
    plt.Figure
        Matplotlib figure containing the plots.
    """

    # Generate the matplotlib figure
    fig = plt.figure(figsize=(16, 12))

    # Generate three (2 left 1 right) axis in the figure
    axis_L1 = fig.add_subplot(2, 2, 1)
    axis_L2 = fig.add_subplot(2, 2, 3)
    axis_R  = fig.add_subplot(1, 2, 2)

    # Set some properties in each of the axes
    axis_L1.set_xlabel(r'$n_s$')
    axis_L1.set_ylabel(r'$L(n_s)$')
    axis_L2.set_xlabel(r'$n_s$')
    axis_L2.set_ylabel(r'$|(\hat{L}(n_s) - L(n_s))|$')
    axis_R.set_xlabel(r'$\omega$')
    axis_R.set_ylabel(r'$\rho(\omega)$')
    axis_L1.grid('#fae1dd', alpha=0.3)
    axis_L2.grid('#fae1dd', alpha=0.3)
    axis_R.grid('#fae1dd',  alpha=0.3)

    # ns values to use in the plots
    ns_vals = torch.arange(0, Lp.shape[1])

    # Plot the coefficients side-by-side.
    axis_L1.bar(ns_vals,       Ll[ex, :].detach(), color=COLOR_1, alpha=1.0, width=0.4)
    axis_L1.bar(ns_vals + 0.4, Lp[ex, :].detach(), color=COLOR_2, alpha=1.0, width=0.4)

    # Calculate difference between labels and predictions of coefficients
    delta_L = Ll[ex, :] - Lp[ex, :]

    # Create color list, blue for positive, red for negative
    cc = [COLOR_1 if val > 0 else COLOR_3 for val in delta_L]

    # Plot the absolute difference between coefficients.
    abs_diff = (Ll[ex, :] - Lp[ex, :]).abs()

    # Plot the bars
    axis_L2.bar(ns_vals, abs_diff.detach(), color=cc, alpha=1.0, width=0.5)

    # Plot the spectral function in the corresponding axes
    axis_R.plot(omega, Rl[ex, :].detach(), color=COLOR_1, linestyle='-',  alpha=1.0)
    axis_R.plot(omega, Rp[ex, :].detach(), color=COLOR_2, linestyle='--', alpha=1.0)

    # Add two rectangles to the handles to show this examples
    handles = [
        (
            pat.Rectangle((0, 0), 2.0, 1.0, color=COLOR_1, alpha=1.0),
            pat.Rectangle((0, 0), 2.0, 1.0, color=COLOR_2, alpha=1.0)
        )
    ]

    # Add a legend to the figure
    fig.legend(
        handles, [f'{ne}: Label / Prediction'], numpoints=1, ncol=1, 
        frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)},
        bbox_to_anchor=(0, 0.95, 1, 0), loc='upper center'
    )

    # Add a fake legend to the L2 to show the colours
    axis_L2.legend(
        [
            pat.Rectangle((0, 0), 2.0, 1.0, color=COLOR_1, alpha=1.0),
            pat.Rectangle((0, 0), 2.0, 1.0, color=COLOR_3, alpha=1.0)
        ],
        ['Positive', 'Negative'], ncol=1, loc='upper right', 
        frameon=False, fontsize='x-small'
    )

    # Delete the copied tensors
    del Ll, Lp, Rl, Rp, abs_diff

    # Return the figure to save it or manipulate it
    return fig

def generate_validate_sets(dataset: SpectralDataset, val_Nb: int, num_sets: int) -> list[SpectralDataset]:
    """ Generate some validation datasets to be used in the trainin """

    # Generate a list that will contain the validation datasets
    validations = [None] * num_sets

    for v in range(num_sets):

        # Generate a validation dataset with the same components as dataset
        val_dataset = SpectralDataset(
            dataset.peak_types, dataset.peak_limits, dataset.kernel, 
            dataset.max_np, dataset.fixed_np, dataset.peak_ids
        )

        # Generate some data in the validation dataset
        val_dataset.generate(val_Nb, dataset.Ns, dataset.U)

        # Add the validation dataset to the list
        validations[v] = val_dataset

    return validations


# TODO: Use DistributedParallel
def train_network(
        network: Network, dataset: SpectralDataset, criterion: torch.nn.Module, train_info: dict, device: torch.device, run_name: str
    ):
    """ Train a neural network module in a given dataset using a loss function.

    --- Parameters:
    network: Network
        Neural network to be trained.
    dataset: SpectralDataset
        Spectral function dataset used in the training.
    criterion: torch.nn.Module
        Loss function used in the training.
    train_info: dict
        Dictionary containing the information about training.
    device: torch.device
        Device where the neural network will be trained.
    run_name: str
        Name of the run. It is used by tensorboard
    """

    # Assert the train_info dictionary contains some keys
    assert all(k in train_info for k in ['epochs', 'batch_size', 'lr_decay', 'val_Nb', 'num_valid'])

    # Generate several validation sets
    validations = generate_validate_sets(dataset, train_info['val_Nb'], train_info['num_valid'])

    # Wrap the given network around DataParallel if possible
    network = torch.nn.parallel.DataParallel(network).to(device)

    # Generate the optimiser used in the training
    sgd_optim = torch.optim.Adam(network.parameters())

    # Modify the learning rate throughout the training
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(sgd_optim, gamma=train_info['lr_decay'])

    # Generate some writers to flush the information
    writer = SummaryWriter(os.path.join('./status/runs', run_name))

    # Generate a writer specific to the train
    writer_train = SummaryWriter(os.path.join('./status/runs', run_name, 'train'))

    # Generate several validation writers
    writer_valid = [
        SummaryWriter(os.path.join('./status/runs', run_name, f'valid_{s}')) \
        for s in range(train_info['num_valid'])
    ]

    # Add the network graph to the writer
    writer.add_graph(network, dataset[:train_info['batch_size']].C)

    # Flush the network data
    writer.flush()

    # Train the network in the given dataset
    for epoch in range(train_info['epochs']):

        # Create a DataLoader object to be used in the training
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=train_info['batch_size'], shuffle=True
        )

        # Add some information to the writer
        writer.add_scalar("train/lr", torch.Tensor(lr_scheduler.get_last_lr()), epoch)

        # Train for each minibatch
        for b, pair in enumerate(train_loader):

            # Move each of the pairs to the network device
            C_label, L_label = pair.C.to(device).log(), pair.L.to(device)

            # Calculate the prediction of the network
            L_hat_train = network(C_label)

            # Calculate the loss function using the prediction of the network
            loss = criterion(L_hat_train, L_label, C_label)

            # Add the loss to the writer
            writer_train.add_scalar("loss",  loss, epoch * len(train_loader) + b)

            # Print every some steps
            if (b + 1) % (int(0.3 * len(train_loader))) == 0:
                print(
                    ' -- Training: Epoch [{}/{}], Step [{},{}], Loss: {:.4f}'.format(
                        epoch + 1, train_info['epochs'], b + 1, len(train_loader), loss.item()
                    ), flush=True
                )

            # Set the gradients to zero in the optimiser
            sgd_optim.zero_grad()

            # Calculate the backward propagation of the loss function
            loss.backward()

            # Update the weights using a gradient descent step
            sgd_optim.step()

            # Delete all uneeded tensor
            del C_label, L_label, L_hat_train

        # Iterate for all validation sets to validate the training
        for nv, valid_set in enumerate(validations):

            # Generate a validation loader to be used in the training
            valid_loader = torch.utils.data.DataLoader(
                valid_set, batch_size=train_info['batch_size'], shuffle=False
            )

            # Get the validation metrics for this set
            valid_loss, total_dL, total_dR = 0.0, 0.0, 0.0

            # Iterate inside the validation dataset
            for nvb, pair in enumerate(valid_loader):

                # Move the pair to the correct device
                C_label, L_label = pair.C.to(device).log(), pair.L.to(device)

                # Compute the validation prediction
                L_hat_valid = network(C_label)

                # Get the validation loss function
                valid_loss += criterion(L_hat_valid, L_label, C_label).item()

                # Compute the MSError in the coefficients
                total_dL += float(((pair.L - L_hat_valid.cpu())  ** 2).mean())
            
                # Compute the difference in the reconstructed spectral function
                R_label     = pair.L[:, 0, :].cpu() @ valid_set.U.T
                R_hat_valid = L_hat_valid.to(R_label.device)[:, 0, :] @ valid_set.U.T

                # Compute the total |max| distance for continuous functions
                total_dR += float((R_label - R_hat_valid).abs().max(axis=1).values.mean())

                # Plot some examples in the last iteration
                if (nvb + 1) == len(valid_loader):

                    # Monitor some examples in the validation set
                    for nf in range(3):

                        # Construct a figure to be saved
                        figure = plot_validate_example(
                            L_label[:, 0, :].detach().cpu(), L_hat_valid[:, 0, :].detach().cpu(), 
                            R_label.detach().cpu(), R_hat_valid.detach().cpu(), valid_set.kernel.omega, epoch, nf
                        )

                        # Add the figure to the writer
                        writer.add_figure(f"train/validate{nv}_ex{nf}", figure, epoch)

                # Delete all uneeded tensors
                del C_label, L_label, L_hat_valid, R_label, R_hat_valid

            # Log some validation information
            print(
                ' -- Validation {}: Epoch [{}/{}], Loss: {:.4f}, dL: {:.4f}, dR: {:.4f}'.format(
                nv, epoch + 1, train_info['epochs'], valid_loss / nvb, total_dL / nvb, total_dR / nvb
                ), flush=True
            )

            # Write the validation loss to the writer
            writer_valid[nv].add_scalar("loss", valid_loss / nvb, (epoch + 1) * len(train_loader))

            # Add the validation dL and dR metrics
            writer_valid[nv].add_scalar("train/dL", total_dL / nvb, epoch)
            writer_valid[nv].add_scalar("train/dR", total_dR / nvb, epoch)

        # Flush the data to the disk
        writer.flush()
        writer_train.flush()
        [w.flush() for w in writer_valid]

        # Modify the learning rate
        lr_scheduler.step()

    # Flush the writer data and close it
    writer.close()
    writer_train.close()
    [w.close() for w in writer_valid]

if __name__ == '__main__':
    pass
