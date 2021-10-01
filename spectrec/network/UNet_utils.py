# Import some builtin modules
from typing import Optional

# Import some third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage(nn.Module):
    """ One dimensional stage of a UNet network. """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, p: float = 0.0):

        # Call the base class constructor
        super().__init__()

        # Some references to keep the code cleaner
        Ci = in_channels
        Co = out_channels
        Cm = out_channels if not mid_channels else mid_channels

        # Generate the module layers
        self.layers = nn.Sequential(
            nn.Dropout(p), nn.Conv1d(Ci, Co, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(Cm), nn.LeakyReLU(inplace=True),
            nn.Dropout(p), nn.Conv1d(Cm, Co, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(Co), nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the module. """
        return self.layers(x)

class DownStage(nn.Module):
    """ One dimensional downstage of a UNet network. """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, p: float = 0.0):
        
        # Call the base class constructor
        super().__init__()

        # Generate the module layers
        self.layers = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            Stage(in_channels, out_channels, mid_channels, p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the module. """
        return self.layers(x)

class UpStage(nn.Module):
    """ One dimensional upstage of a UNet network. """

    def __init__(self, in_channels: int, out_channels: int, p: float = 0.0):

        # Call the base class constructor
        super().__init__()

        self.Ci, self.Co = in_channels, out_channels

        # Generate the layer to upsample the data and halve channels
        self.up = nn.Sequential(
            nn.Upsample(mode = 'linear', scale_factor = 2, align_corners = True),
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )

        # Generate the convolutional stage layer
        self.conv = Stage(in_channels, out_channels, in_channels // 2, p)

    def forward(self, x_up: torch.Tensor, x_crop: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the module. """

        # Upsample the tensor from the previous stage
        x_up = self.up(x_up)

        # Crop x_crop around the middle with the length of x_up
        x_crop = UpStage.__crop_center(x_crop, x_up.shape[2])

        # Concatenate both tensors along the channels direction
        x = torch.cat([x_crop, x_up], dim = 1)

        return self.conv(x)

    @staticmethod
    def __crop_center(tensor: torch.Tensor, num_points: int) -> torch.Tensor:
        """ Crop num_points elements from a tensor around the middle. """
        # Obtain the midpoint of the tensor along the L direction
        mid_point = tensor.size(2) // 2
        beg_point = mid_point - num_points // 2
        end_point = mid_point + num_points // 2
        return tensor[:, :, beg_point:end_point]

class CNNOut(nn.Module):
    """ Convolutional output of a 1d UNet network. """

    def __init__(self, in_channels: int, out_channels: int, p: float = 0.0):

        # Call the base class constructor
        super().__init__()

        # Generate the layers of the module
        self.conv = nn.Sequential(
            nn.Dropout(p), nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the module. """
        return self.conv(x)

class FCOut(nn.Module):
    """ Fully connected output of a one dimensional UNet network. """

    def __init__(self, input_size: int, output_size: int, p: float = 0.0):

        # Call the base class constructor
        super().__init__()

        # Generate the layers of the module
        self.fc_out  = nn.Sequential(
            nn.Dropout(p), nn.Linear(input_size, 256), nn.LeakyReLU(inplace=True),
            nn.Dropout(p), nn.Linear(256, 128), nn.LeakyReLU(inplace=True),
            nn.Dropout(p), nn.Linear(128, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the module. """
        return self.fc_out(x)

if __name__ == '__main__':
    pass
