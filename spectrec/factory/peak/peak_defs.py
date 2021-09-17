# Import some builtin modules
from typing import Union, Sequence, Optional
from functools import lru_cache

# Some third party modules
import numpy as np
import torch

# Load the Peak class to be used as interface
from .Peak import Peak

class GaussianPeak(Peak):
    ''' Gaussian peak implementation. '''

    # Name of the peak as a string, shared among all instances
    _peak_type: str        = 'GaussianPeak'
    _param_ids: tuple[str] = ('M', 'A', 'W')

    @lru_cache(maxsize = 2)
    def generate(self, omega: torch.Tensor) -> torch.Tensor:
        """ Evaluate the Peak in the given energy range. The Peak is evaluated
        using the standard Gaussian formula.
                    
                    A * exp(- (omega - M) ** 2 / (2 * W ** 2))

        --- Parameters
        omega: torch.Tensor
            Energy range where the peak will be evaluated

        --- Returns
        torch.Tensor
            Evaluation of the Peak in the given energy range
        """
        return self.A * torch.exp(- (omega - self.M) ** 2 / (2 * self.W ** 2))

    @property
    def M(self) -> float:
        ''' Mass of the peak. '''
        return self._param_values['M']

    @property
    def A(self) -> float:
        ''' Amplitude of the peak. '''
        return self._param_values['A']

    @property
    def W(self) -> float:
        ''' Width of the peak. '''
        return self._param_values['W']

class DeltaPeak(Peak):
    ''' Dirac delta peak implemetation. '''

    # Name of the peak as a string, shared among all instances
    _peak_type: str        = 'DeltaPeak'
    _param_ids: tuple[str] = ('M', 'A')

    def generate(self, omega: torch.Tensor) -> torch.Tensor:
        ''' Generate a numpy representation of the peak in the energy range. '''

        # Step size in omega
        dw = float(omega[-1] - omega[0])

        # Transform the mass into an index
        M_idx = int((self.M - float(omega[0])) * (omega.shape[0] / dw))

        # Generate an array of zeros
        delta = np.zeros(omega.shape[0])

        # Change index corresponding to the mass with the correct amplitude
        delta[M_idx] = self.A

        return delta

    @property
    def M(self) -> float:
        """ Mass of the peak. """
        return self._param_values['M']

    @property
    def A(self) -> float:
        """ Amplitude of the peak. """
        return self._param_values['A']

if __name__ == '__main__':
    pass
