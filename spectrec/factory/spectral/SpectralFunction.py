# -- Import built-in modules
from typing import Optional, Sequence
import random

# -- Import third-party modules
import matplotlib.pyplot as plt
import torch

# Import several abstract classes
from spectrec.factory import Peak
from spectrec.factory import Kernel


class SpectralFunction:
    """ Spectral function object: composed of several peaks. """

    def __init__(
        self, peak_types: list[Peak], peak_limits: dict[str, Sequence[float]], kernel: Kernel, 
        max_np: int, fixed_np: bool = True, peak_ids: Optional[list] = None
    ):

        # Save the peak types used in the calculation and the limits
        self.peak_types, self.peak_limits = peak_types, peak_limits

        # Save the maximum number of peaks and whether they are fixed or not
        self.max_np, self.fixed_np = max_np, fixed_np

        # Save the kernel in the object
        self.kernel = kernel

        # Save the peak_ids depending of the input passed
        if peak_ids:
            # If peak_ids is passed, then it should have the same dimensions as peak_types
            assert len(peak_ids) == len(peak_types), 'There must be a peak if for each peak type'

            # Save the peak_id in the class
            self.peak_ids = peak_ids
        else:
            # If it is not passed, then peak_ids is s list of None
            self.peak_ids = [None] * len(peak_types)

        # Sample some random peaks using the information provided
        self.generate_random_peaks()

        # Variables where the actual data will be stored. Used to optise computation
        self.__data = {'R': None, 'C': None}

    def generate_random_peaks(self):
        r""" Sample several peaks from the available peak classes saved in the object
        (self.peak_types). The number of peaks sampled varies depending on the parameters
        self.max_np and self.fixed_np. If self.fixed_np is set to True, then the number of peaks is
        always self.max_np. In the case in which self.fixed_np = False, then the number of peaks
        sampled is a random integer in [1, self.max_np]. The peak limits are created using the
        limits contained in self.peak_limits.

        Example: Generate a spectral function composed by a maximum of three peaks sampling randomly
        from three different peak types: a GaussianPeak, a DeltaPeak and a background GaussianPeak.
        Then, the properties of the spectral function should be:

        self.peak_types  = [GaussianPeak, DeltaPeak, GaussianPeak]
        self.peak_limits = Dictionary of limits
        self.max_np      = 3
        self.fixed_np    = False
        self.peak_ids    = [None, None, 'b']

        and self.peak_limits could be for example
        {
            'A':  [0.1, 1.0], 'M':  [0.1, 3.5], 'W':  [0.01, 0.05],
            'Ab': [0.5, 1.0], 'Mb': [4.0, 7.0], 'Wb': [0.10, 0.20],
        }

        The code will randomly select up to three different peaks types from self.peak_types. For
        example, if it selects two peak types and they correspond to the first GaussianPeak and the
        second GaussianPeak, then the first one will use the first row of limits in
        self.peak_limits while the second one will use the second row of limits in the
        dictionary of limits. In the case in which DeltaPeak is sampled, it would use the first row
        of limits in the dictionary.
        """

        # Select the number of peaks to tbe sampled in the function
        sel_np = self.max_np if self.fixed_np else random.randint(1, self.max_np)

        # List that will contain the peaks in the spectral function
        self.peaks = [None] * sel_np

        # Iterate several times to sample different peaks
        for p in range(sel_np):

            # Select a random example from the list
            sp = random.randint(0, len(self.peak_types) - 1)

            # Instantiate the peak_class using the limits
            self.peaks[p] = self.peak_types[sp](self.peak_limits, self.peak_ids[sp])

    def compute_R(self, device: torch.device = 'cpu', recalculate: bool = False) -> torch.Tensor:
        r""" Compute the tensor representation of the spectral function. The spectral function is
        computed using the peaks included in self.peaks and the information inside the kernel.
        Changing some values of the kernel would lead to a change in the spectral function.

        --- Parameters:
        device: torch.device = 'cpu'
            Device where the spectral function data will be stored.
        recalculate: bool = False
            Force the recalculation of the spectral function. If new peaks are sampled, then the
            spectral function should be recalculated once to update its representation.

        --- Returns
        torch.Tensor
            Tensor containing the spectral function representation: R(w)
        """ 
        # Avoid calculating the spectral function several times if not needed
        if self.__data['R'] is None or recalculate:

            # Torch array where the spectral function will be stored
            rho = 1e-4 * torch.ones(self.kernel.Nw)

            # Add each peak to the spectral function using the kernel
            for peak in self.peaks:
                rho += peak.generate(self.kernel.omega)

            # Integrate the spectral function to obtain \int \rho(w) dw = 6.0
            int_rho = (rho * self.kernel.dw).sum()

            # Normalise the spectral function accordingly
            self.__data['R'] = (rho * 6.0) / int_rho

        return self.__data['R'].to(device)

    def compute_C(self, device: torch.device = 'cpu', recalculate: bool = False) -> torch.Tensor:
        r""" Compute the tensor representation of the correlation function associated to the spectral
        function. The correlation function is computed using:

                            C(\tau) = \int d\omega K(t, w) \rho(w),

        where K(t, w) is the kernel used.

        --- Parameters:
        device: torch.device = 'cpu'
            Device where the correlation function data will be stored.
        recalculate: bool = False
            Force the recalculation of the correlation function. If new peaks are sampled, then the
            spectral function should be recalculated once to update its representation.

        --- Returns
        torch.Tensor
            Tensor containing the correlation function representation: C(t)
        """ 

        # Calculate the correlation function only once
        if self.__data['C'] is None or recalculate:

            # Assert the kernel data has the needed dimensions
            assert self.kernel.kernel.shape == (self.kernel.Nt, self.kernel.Nw)

            # Calculate the correlation function
            self.__data['C'] = self.kernel.kernel.to(device) * self.compute_R(device, recalculate)

            # Integrate over the energy space (columns)
            self.__data['C'] = torch.sum(self.__data['C'] * self.kernel.dw, axis = 1)

        return self.__data['C'].to(device)

    def plot_tensor_representation(self, recalculate: bool = True) -> plt.Figure:
        """ Plot the spectral function tensor representation. """

        # Generate a figure where the data will be plotted
        fig = plt.figure(figsize=(16, 12))

        # Add two axis to the figure, one for C and another for R
        axis_R = fig.add_subplot(1, 2, 1)
        axis_C = fig.add_subplot(1, 2, 2)

        # Add some properties to the axis
        axis_R.set_xlabel(r'$\omega$')
        axis_R.set_ylabel(r'$\rho(\omega)$')
        axis_C.set_xlabel(r'$\tau$')
        axis_C.set_ylabel(r'$C(\tau)$')
        axis_R.grid('#343a40', alpha=0.2)
        axis_C.grid('#343a40', alpha=0.2)

        # The correlation function axis should be in log scale
        axis_C.set_yscale('log')

        # Add the peaks to the title
        title = ''
        for ip, peak in enumerate(self.peaks):
            if ip % 2 != 1:
                title += str(peak) + ' '
            else:
                title += str(peak) + '\n'

        # Set the title of the picture as the peaks
        fig.suptitle(title)

        # Add the axis to the data
        axis_R.plot(self.kernel.omega, self.compute_R(recalculate=recalculate), color='#84a98c')
        axis_C.plot(self.kernel.tau,   self.compute_C(recalculate=recalculate), color='#014f86')

        return fig

    # -- Magic methods of the class {{{
    def __str__(self) -> str:
        return '<SpectralFunction>'

    def __repr__(self) -> str:
        return self.__str__()
    # -- }}}

        
if __name__ == '__main__':
    pass
