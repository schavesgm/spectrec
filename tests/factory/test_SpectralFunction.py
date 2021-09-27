# Import some built-in modules
import re

# Import third party modules
import pytest
import torch

# Import user-defined modules
from spectrec.factory import SpectralFunction
from spectrec.factory import GaussianPeak
from spectrec.factory import DeltaPeak
from spectrec.factory import NRQCDKernel

# -- Global variables to be used in the testing {{{
# Limits to be used in the peaks
limits = {
    'M':  [0.1, 3.5], 'A':  [0.1, 1.0], 'W':  [0.1, 0.2],
    'Mb': [4.0, 6.0], 'Ab': [0.1, 0.3], 'Wb': [0.4, 0.6],
}

# Peak classes to be used in the spectral function
peak_types = [GaussianPeak, DeltaPeak, GaussianPeak]

# Peak ids to determine each of the peaks; the third one is background
peak_ids = [None, None, 'b']

# Generate the NRQCD kernel in the calculation
nrqcd = NRQCDKernel(64, 1000, [0.0, 8.0])
# -- }}}

@pytest.fixture(scope = "session")
def spfunc():
    """ Generate a spectral function to be tested. """
    yield SpectralFunction(peak_types, limits, nrqcd, 3, peak_ids = peak_ids)

class TestSpectralFunction:
    """ Class used to test the SpectralFunction module. """

    def test_initialiser(self, spfunc):
        """ Test the initialiser of the class. """
        assert spfunc.peak_types == peak_types
        assert spfunc.peak_ids   == peak_ids
        assert spfunc.kernel     == nrqcd
        assert spfunc.max_np     == 3
        assert spfunc.fixed_np   == True

    def test_generation_cpu(self, spfunc):
        """ Test the generation of the data is correct. """

        # Generate several spectral functions, as it is random
        for p in range(1000):

            # First, sample some new peaks
            spfunc.generate_random_peaks()

            # Assert the sampled peaks are correct size
            assert all(isinstance(p, (GaussianPeak, DeltaPeak)) for p in spfunc.peaks)
            assert len(spfunc.peaks) == 3

            # Compute the spectral function and the correlation function
            R = spfunc.compute_R(recalculate = True)
            C = spfunc.compute_C(recalculate = True)

            # Assert the sizes are correct
            assert R.shape == torch.Size([spfunc.kernel.Nw])
            assert C.shape == torch.Size([spfunc.kernel.Nt])

            # Assert the integral of R is close to 6
            assert float(C[0]) == pytest.approx(6.0)
            assert float((R * spfunc.kernel.dw).sum()) == pytest.approx(6.0)

    @pytest.mark.skipif('not torch.cuda.is_available()', reason = 'CUDA is not available')
    def test_generation_gpu(self, spfunc):
        """ Test the generation of the data is correct. """

        # Generate several spectral functions, as it is random
        for p in range(1000):

            # First, sample some new peaks
            spfunc.generate_random_peaks()

            # Assert the sampled peaks are correct size
            assert all(isinstance(p, (GaussianPeak, DeltaPeak)) for p in spfunc.peaks)
            assert len(spfunc.peaks) == 3

            # Compute the spectral function and the correlation function
            R = spfunc.compute_R(device = 'cuda', recalculate = True)
            C = spfunc.compute_C(device = 'cuda', recalculate = True)

            # Assert the cuda data is in the correct device
            assert re.match(r'cuda.*$', str(R.device))
            assert re.match(r'cuda.*$', str(C.device))

            # Assert the sizes are correct
            assert R.shape == torch.Size([spfunc.kernel.Nw])
            assert C.shape == torch.Size([spfunc.kernel.Nt])

            # Assert the integral of R is close to 6
            assert float(C[0]) == pytest.approx(6.0)
            assert float((R.to('cpu') * spfunc.kernel.dw).sum()) == pytest.approx(6.0)

if __name__ == '__main__':
    pass
