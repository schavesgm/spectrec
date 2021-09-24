# Import some built-in methods
import os
from collections import OrderedDict
from typing import Union

# Import some third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import some user defined modules
from .UNet_utils import Stage
from .UNet_utils import DownStage
from .UNet_utils import UpStage
from .UNet_utils import CNNOut
from .UNet_utils import FCOut

class UNet(nn.Module):
    """ One dimensional convolutional neural network. """

    def __init__(self, input_size: int, output_size: int, in_channels: int = 1, out_channels: int = 1, p: float = 0.0):
        
        # Call the base class constructor
        super().__init__()

        # Save the input size of the data in the network
        self.Li, self.Lo = input_size, output_size

        # Save the number of input and output channels
        self.Ci, self.Co = in_channels, out_channels

        # Amplify the data using a linear layer if needed
        if self.Li < 572:
            self.pre_amplify = nn.Sequential(
                nn.Linear(input_size, 572, bias = False), nn.ReLU(inplace = True)
            )

        # Generate the network architecture
        self.initial = Stage(in_channels, 64, p)
        self.down_1  = DownStage(64, 128, p)
        self.down_2  = DownStage(128, 256, p)
        self.down_3  = DownStage(256, 512, p)
        self.down_4  = DownStage(512, 1024, p) 
        self.up_1    = UpStage(1024, 512, p)
        self.up_2    = UpStage(512, 256, p)
        self.up_3    = UpStage(256, 128, p)
        self.up_4    = UpStage(128, 64, p)
        self.cnn_out = CNNOut(64, out_channels, p)

        # Provided the number of output channels is one, then FCout to (Nb, Lo)
        if out_channels == 1:
            self.fc_out = FCOut(388, output_size, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the UNet network. """

        # In case the input data is smaller than 572, use the preamplifier
        if self.Li < 572:
            x = self.pre_amplify(x)

        # Initial stage of the network
        x_d1 = self.initial(x)

        # Downpass (Encoder) part of the network
        x_d2 = self.down_1(x_d1)
        x_d3 = self.down_2(x_d2)
        x_d4 = self.down_3(x_d3)
        x_d5 = self.down_4(x_d4)

        # Uppass (Decoder) part of the network
        x = self.up_1(x_d5, x_d4)
        x = self.up_2(x, x_d3)
        x = self.up_3(x, x_d2)
        x = self.up_4(x, x_d1)

        # Convolutional network output
        x = self.cnn_out(x)

        # Apply the fully connected layer if output channels is 1
        if self.Co == 1:

            # Do the forward pass on the fully connected
            x = self.fc_out(x.view(x.shape[0], -1))

            # Transform the fully connected to (Nb, 1, Ns)
            x = x.view(x.shape[0], 1, x.shape[1])

        return x

    def set_params(self, state_dict: dict, eliminate_str: str = 'module.') -> None:
        """ Set the parameters of the model using the state_dict provided. """

        # Generate a new dictionary to clean the keys
        clean_state_dict: dict[str, torch.Tensor] = {}

        # Iterate to clean the data
        for key, value in state_dict.items():
            clean_state_dict[key.replace(eliminate_str, '', 1)] = value
        
        self.load_state_dict(OrderedDict(clean_state_dict))

    def save_params(self, name: str, path: str = './status/net') -> None:
        """ Save the current status of the network into the given file. """

        # Check if the path exists
        if not os.path.exists(path): os.makedirs(path)

        # Save the weights into the path
        torch.save(self.state_dict(), os.path.join(path, name + '.pt'))

    def load_params(self, name: str, path: str = './status/net', device: Union[torch.device, str] = 'cpu', verbose: bool = True) -> None:
        """ Load the status of a network with given name. """
        
        # Path to the given file
        path = os.path.join(path, name + '.pt')

        try:
            # Load the parameters
            self.load_state_dict(torch.load(path, map_location = device))

            # Evaluate the parameters in the model
            self.eval()

            if verbose: print(f'Loaded {name} UNet instance', flush = True)

        except FileNotFoundError:
            if verbose: print(f'Network with name {name} not found. New instance', flush = True)

    @property
    def num_weights(self) -> int:
        """ Show the total number of learnable parameters. """
        return sum(p.numel() for p in self.parameters())

if __name__ == '__main__':
    pass
