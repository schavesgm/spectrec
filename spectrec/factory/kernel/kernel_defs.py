# -- Import built-in modules
from functools import lru_cache

# -- Import some third party modules
import torch

# -- Import user-defined modules
from .Kernel import Kernel

class NRQCDKernel(Kernel):
    """ Non-relativistic QCD kernel definition. """

    # Identifier for this kernel type
    _kernel_type: str = 'NRQCDKernel'

    @lru_cache(maxsize = 1)
    def _calculate_kernel(self, Nt: int, Nw: int, w_min: float, w_max: float) -> torch.Tensor:
        return (- self.tau.view(self.Nt, 1) * self.omega.view(1, self.Nw)).exp()

if __name__ == '__main__':
    pass
    
