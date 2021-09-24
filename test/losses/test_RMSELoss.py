# Import some third-party modules
import pytest
import torch

# Import some user-defined modules
from spectrec.losses  import RMSELoss
from spectrec.factory import GaussianPeak
from spectrec.factory import NRQCDKernel
from spectrec.factory import SpectralDataset

# from spectrec.spfactory.kernel   import NRQCDKernel
# from spectrec.spfactory.peak     import GaussianPeak
# from spectrec.spfactory.set      import SpectralSet

@pytest.fixture(scope = "session")
def kernel():
    yield NRQCDKernel(64, 1000, [0.0, 8.0])

@pytest.fixture(scope = "session")
def dataset(kernel):

    # Limits used to define the peaks
    limits = {'M': [0.5, 1.0], 'A': [0.2, 1.0], 'W': [0.1, 0.2]}

    # Generate the spectral dataset
    yield SpectralDataset([GaussianPeak], limits, kernel, 3)

class TestRMSELoss:
    ''' Class used to test the RMSELoss class. '''

    def test_behaviour(self, dataset, kernel):
        ''' Test the behaviour of the RMSELoss class. '''

        # Generate the data using the dataset
        dataset.generate(2000, 64)

        # Some references to the data in the dataset
        in_data, U = dataset[:], dataset.U

        # Generate several RMSELoss objects to test them
        rmseloss_1 = RMSELoss(1.0, 1.0, kernel, U) 
        rmseloss_2 = RMSELoss(0.0, 1.0, kernel, U) 
        rmseloss_3 = RMSELoss(1.0, 0.0, kernel, U) 

        # Calculate the losses
        loss_1 = rmseloss_1(in_data.L, in_data.L, in_data.C.log())
        loss_2 = rmseloss_1(in_data.L, in_data.L, in_data.C.log())
        loss_3 = rmseloss_1(in_data.L, in_data.L, in_data.C.log())

        # Assert that the results are close to zero
        assert float(loss_1) == pytest.approx(0.0, abs = 0.1)
        assert float(loss_2) == pytest.approx(0.0, abs = 0.1)
        assert float(loss_3) == pytest.approx(0.0, abs = 0.1)

if __name__ == '__main__':
    pass
