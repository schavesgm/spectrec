# Import third-party modules
import pytest

# Import some needed classes and functions
from spectrec.factory import Peak
from spectrec.factory import Kernel
from spectrec.network import Network

from spectrec.factory import NRQCDKernel
from spectrec.factory import GaussianPeak
from spectrec.factory import DeltaPeak
from spectrec.network import UNet

from spectrec.utils import register_kernel_class
from spectrec.utils import register_peak_class
from spectrec.utils import register_network_class
from spectrec.utils import retrieve_kernel_class
from spectrec.utils import retrieve_peak_class
from spectrec.utils import retrieve_network_class

class TestNPKFactory:

    def test_peak_factory(self):
        """ Test the peak factory functions. """

        # Register some peaks in the class
        register_peak_class('GaussianPeak', GaussianPeak)

        # Retrieve the class and assert is subclass
        assert issubclass(retrieve_peak_class('GaussianPeak'), Peak)

        # Try registering a kernel into the peaks
        with pytest.raises(AssertionError) as info:
            register_peak_class('FakePeak', NRQCDKernel)

        # Assert an error is raised when adding a false peak
        assert 'must be a subclass of Peak' in str(info.value)

        # Try obtaining a peak that is not valid
        with pytest.raises(KeyError) as info:
            retrieve_peak_class('DeltaPeak')

        # Assert an error is raised when retrieving an unregistered peak
        assert 'is not registered' in str(info.value)

    def test_kernel_factory(self):
        """ Test the kernel factory functions. """

        # Register some kernels in the class
        register_kernel_class('NRQCDKernel', NRQCDKernel)

        # Retrieve the class and assert is subclass
        assert issubclass(retrieve_kernel_class('NRQCDKernel'), Kernel)

        # Try registering a peak into the kernels
        with pytest.raises(AssertionError) as info:
            register_kernel_class('FakePeak', DeltaPeak)

        # Assert an error is raised when adding a false kernel
        assert 'must be a subclass of Kernel' in str(info.value)

        # Try obtaining a kernel that is not valid
        with pytest.raises(KeyError) as info:
            retrieve_kernel_class('FakeKernel')

        # Assert an error is raised when retrieving an unregistered kernel
        assert 'is not registered' in str(info.value)

    def test_network_factory(self):
        """ Test the network factory functions. """

        # Register a network in the class
        register_network_class('UNet', UNet)

        # Retrieve the class and assert is subclass
        assert issubclass(retrieve_network_class('UNet'), Network)

        # Try registering a peak into the networks
        with pytest.raises(AssertionError) as info:
            register_network_class('FakeNet', DeltaPeak)

        # Assert an error is raised when adding a false network
        assert 'must be a subclass of Network' in str(info.value)

        # Try obtaining a network that is not valid
        with pytest.raises(KeyError) as info:
            retrieve_network_class('FakeNet')

        # Assert an error is raised when retrieving an unregistered network
        assert 'is not registered' in str(info.value)


if __name__ == '__main__':
    pass

