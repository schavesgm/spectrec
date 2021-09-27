# Load some builtin modules
from typing import Optional

# Load some third party modules
import torch
import torch.nn as nn

# Load some needed spectrec classes
from spectrec.factory import Kernel

class MSELoss(nn.Module):
    r""" 
        MSE = (Nb)^{-1} \sum_{b=1}^{Nb} \sum_{s=1}^{ns} (y_{bs} - \hat{y}_{bs})^2
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average: bool = True):
        super().__init__()

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, *args) -> torch.Tensor:
        """ Forward pass of the loss function. """
        return ((outputs - labels) ** 2).sum(dim = 2).mean()

class RMSELoss(nn.Module):
    """ Recurrent MSELoss, the loss function is the sum of two losses functions. The first
    one is the loss between the predicted and label coefficients. The second one is the loss
    between the label correlation function and the reconstructed correlation function using
    the predicted coefficients and the kernel.
    """

    def __init__(self, a: float, b: float, kernel: Kernel, basis: torch.Tensor, weight: Optional[torch.Tensor] = None, size_average: bool = True):
        """ Constructor of the class:

        --- Parameters:
        a: float
            Weight for the loss between predicted and label coefficients.
        b: float
            Weight for the loss between the label correlation function and the reconstructed
            correlation function using the kernel and the basis.
        kernel: Kernel
            Kernel used in the dataset.
        basis: torch.Tensor
            Basis functions used in the dataset.
        """
        super().__init__()

        # Save some needed quantities needed in the calculation
        self.a, self.b, self.dw, self.kernel, self.basis = a, b, kernel.dw, kernel.kernel, basis
    
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the loss function. Inputs must be the logarithm of the correlation function. """

        # Get the device where the outputs are allocated
        device = outputs.device

        # Move the kernel and the basis to the correct device
        self.kernel = self.kernel.to(device) if self.kernel.device != device else self.kernel
        self.basis  = self.basis.to(device)  if self.basis.device  != device else self.basis

        # Get some references to the number of batches and time points
        Nb, Nt = outputs.shape[0], inputs.shape[2]

        # First, calculate the spectral function from the output
        output_R = outputs @ self.basis.T

        # Integrate the spectral function to obtain \int \rho(w) dw = 6.0
        int_rho = (output_R * self.dw).sum(dim = 2)

        # Normalise the spectral function accordingly
        output_R = torch.divide((output_R * 6.0), int_rho.view(Nb, 1, 1))

        # Now, integrate the spectral function using the kernel to obtain C
        output_C = torch.reshape(
            (self.kernel * output_R * self.dw).sum(dim = 2), (Nb, 1, Nt)
        )

        # Take the log of the positive correlation function
        output_C = output_C.abs().log()

        # Calculate the mean squared error for both outputs
        mse_outputs = ((outputs  - labels) ** 2).sum(dim = 2).mean()
        mse_inputs  = ((output_C - inputs) ** 2).sum(dim = 2).mean()

        return self.a * mse_outputs + self.b * mse_inputs

if __name__ == '__main__':
    pass
