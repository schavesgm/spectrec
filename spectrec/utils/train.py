# Load some built-in modules
import os

# Load some third-party modules
import torch
import matplotlib.pyplot as plt

# Load some user-defined modules
from spectrec.factory import SpectralDataset
from .plots import eliminate_mirror_axis

def train_network(
    network: torch.nn.Module, dataset: SpectralDataset, loss: torch.nn.Module, 
    epochs: int, device: torch.device, batch_size: int, path: str = './status/monitor/train'
    ):
    """ Train a neural network module in a given dataset using a loss function.

    --- Parameters:
    network: torch.nn.Module
        Neural network to be trained.
    dataset: SpectralDataset
        Spectral function dataset used in the training. 
    loss: torch.nn.Module
        Loss function used in the training.
    epochs: int
        Number of epochs used in the training.
    device: torch.device
        Device where the neural network will be trained.
    batch_size: int
        Number of batches to simultaneously train a network.
    path: str
        Path where the monitor data will be stored.
    """

    # Wrap the given network in a DataParallel if possible
    net_train = torch.nn.DataParallel(network).to(device)

    # Generate an optimiser to train the network
    gd_optimiser = torch.optim.Adam(net_train.parameters())

    # Modify the learning rate throughout the training
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gd_optimiser, gamma = 0.9)

    # Generate the path where the train monitoring data will be stored
    path = os.path.join(path, dataset.name)

    # If the path is not created, then create a directory
    if not os.path.exists(path): os.makedirs(path)

    # Create a buffer where the data will be flushed
    stream_out = open(os.path.join(path, 'monitor.dat'), 'w')

    # List where the monitoring data will be plotted
    loss_values, loss_indices = [], []

    # Train the network in the given dataset
    for epoch in range(epochs):

        # Create a DataLoader object to be used in the training
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size, shuffle = True
        )

        for b, pair in enumerate(train_loader):

            # Move each of the pairs to the network device
            C_label, L_label = pair.C.to(device).log(), pair.L.to(device)

            # Calculate the prediction of the network
            L_pred = net_train(C_label)

            # Calculate the loss function using the prediction of the network
            loss_value = loss(L_pred, L_label, C_label)

            # Set the gradients to zero in the optimiser
            gd_optimiser.zero_grad()

            # Calculate the backward propagation of the loss function
            loss_value.backward()

            # Update the weights using a gradient descent step
            gd_optimiser.step()
            
            # Flush some data into the stream every few iterations
            if b % 20 == 0:
                # Flush some data to a file and the console
                stream_out.write(f'{epoch} {b} {float(loss_value)}\n')
                print(epoch, b, loss_value, flush = True)

                # Append some data to the lists to generate some plots
                loss_values.append(float(loss_value))
                loss_indices.append(epoch * len(train_loader) + b)

            # Delete all uneeded tensor
            del C_label, L_label, L_pred

        # Modify the learning rate
        lr_scheduler.step()

    # Close the stream
    stream_out.close()

    # Generate a figure to plot the data
    fig = plt.figure(figsize = (16, 10))

    # Add an axes to the figure
    axis = fig.add_subplot(1, 1, 1)

    # Add some information to the axis
    axis.set_xlabel(r'$e \cdot N_b + b$')
    axis.set_ylabel(r'Loss')
    axis.grid('#fae1dd')
    axis.set_facecolor('#fcfcfc')

    # Plot the data
    axis.plot(loss_indices, loss_values, color = '#eb5e28')

    # Eliminate the mirror axis from the figure
    eliminate_mirror_axis(axis)

    # Save the figure
    fig.savefig(os.path.join(path, 'train_loss.pdf'))

    # Set the parameters in the non-parallel network
    network.set_params(net_train.state_dict())

if __name__ == '__main__':
    pass
