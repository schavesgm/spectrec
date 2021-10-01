# Import some built-in modules
import os
import shutil

# Import some third-party modules
import pytest
import torch

# Import some user-defined modules
from spectrec.factory import SpectralDataset
from spectrec.factory import GaussianPeak
from spectrec.factory import NRQCDKernel

# -- Global variables to be used in the fit {{{
# Limits used in the peaks
limits = {'M': [0.1, 4.0], 'A': [0.1, 1.0], 'W': [0.5, 0.2]}

# Peak types to be used in the set
peak_types = [GaussianPeak]

# Generate a Kernel objects to test the class
nrqcd = NRQCDKernel(64, 1000, [0.0, 8.0])
# -- }}}

@pytest.fixture(scope = "session")
def dataset():
    """ Generate a dataset to be tested. """
    yield SpectralDataset(peak_types, limits, nrqcd, 3)

class TestSpectralDataset:

    def test_initialiser(self, dataset):
        """ Test the initialiser of the class. """

        assert dataset.peak_types == peak_types
        assert dataset.peak_ids   == [None]
        assert dataset.kernel     == nrqcd
        assert dataset.max_np     == 3
        assert dataset.fixed_np   == True

    def test_generate(self, dataset):
        """ Test the correct initialisation of the data. """

        # Generate a dataset of 2000 examples
        dataset.generate(2000, 96)

        # Assert some conditions on the newly created dataset
        assert dataset.is_generated
        assert dataset.Nb == 2000
        assert dataset.Ns == 96
        assert dataset.R.shape == torch.Size([2000, 1000])
        assert dataset.C.shape == torch.Size([2000, 64])
        assert dataset.L.shape == torch.Size([2000, 96])
        assert dataset.U.shape == torch.Size([1000, 96])

        # Assert R can be reconstructed with some accuracy using L
        diff = (dataset.R - dataset.L @ dataset.U.T).mean()
        assert diff == pytest.approx(0.0, abs = 0.001)

        # Generate another dataset using the first basis functions
        dataset.generate(3000, 96, basis = dataset.U)

        # Assert some conditions on this dataset
        assert dataset.is_generated
        assert dataset.Nb == 3000
        assert dataset.Ns == 96
        assert dataset.R.shape == torch.Size([3000, 1000])
        assert dataset.C.shape == torch.Size([3000, 64])
        assert dataset.L.shape == torch.Size([3000, 96])
        assert dataset.U.shape == torch.Size([1000, 96])

        # Assert R can be reconstructed with some accuracy using L
        diff = (dataset.R - dataset.L @ dataset.U.T).mean()
        assert diff == pytest.approx(0.0, abs = 0.001)

    def test_clear(self, dataset):
        """ Test the dataset can be cleared. """

        # Generate a dataset of 2000 examples
        dataset.generate(2000, 96)

        # Clean the dataset
        dataset.clear()

        # Assert all data is cleaned
        assert dataset.R == None
        assert dataset.C == None
        assert dataset.L == None
        assert dataset.U == None

    def test_save_and_load_dataset(self, dataset):
        """ Test the data can be saved. """

        # Generate a dataset of 2000 examples
        dataset.generate(2000, 96)

        # Save the data in the current path
        dataset.save_dataset('test', 'dataset', 'temp')

        # Path where the data is saved
        path = os.path.join('./temp', f'test_{dataset.name}_dataset')

        # Assert the file exists
        assert os.path.exists(path)

        # List all files inside the directory
        files = os.listdir(path)

        # Assert that some files are present
        assert all(f'{t}_test_{dataset.name}_dataset.dat' in files for t in ['R', 'C', 'L', 'U'])

        # Load the saved files into the dataset
        dataset.load_dataset(2000, 96, 'test', 'dataset', 'temp')

        # Assert the loaded data has the correct dimensions
        assert dataset.Nb == 2000
        assert dataset.Ns == 96 # Remove the temporal directory
        shutil.rmtree('./temp')

    def test_get_subdataset(self, dataset):
        """ Test that we can obtain a subdataset of the set. """

        # Generate a dataset of 2000 examples
        dataset.generate(2000, 96)

        # Get two subdatasets from the dataset
        sub1, sub2 = dataset.get_subdataset(0.5, False), dataset.get_subdataset(0.7, True)

        # Assert some conditions on the subdatasets
        assert isinstance(sub1, SpectralDataset)
        assert isinstance(sub2, SpectralDataset)

        # Assert some conditions on the first dataset
        assert sub1.Nb == dataset.Nb // 2
        assert torch.equal(sub1[:].C, dataset[:1000].C)
        assert torch.equal(sub1[:].L, dataset[:1000].L)

        # Assert some conditions on the second dataset
        assert sub2.Nb == int(0.7 * dataset.Nb)


if __name__ == '__main__':
    pass
