# Import third-party modules
import pytest

# Import some needed classes and functions
from spectrec.factory import Peak
from spectrec.factory import Kernel
from spectrec.factory import NRQCDKernel
from spectrec.factory import GaussianPeak
from spectrec.factory import DeltaPeak

from spectrec.utils import register_kernel_class
from spectrec.utils import register_peak_class
from spectrec.utils import retrieve_kernel_class
from spectrec.utils import retrieve_peak_class

class TestPKFactory:

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

        # Register some peaks in the class
        register_kernel_class('NRQCDKernel', NRQCDKernel)

        # Retrieve the class and assert is subclass
        assert issubclass(retrieve_kernel_class('NRQCDKernel'), Kernel)

        # Try registering a kernel into the peaks
        with pytest.raises(AssertionError) as info:
            register_kernel_class('FakePeak', DeltaPeak)

        # Assert an error is raised when adding a false peak
        assert 'must be a subclass of Kernel' in str(info.value)

        # Try obtaining a peak that is not valid
        with pytest.raises(KeyError) as info:
            retrieve_kernel_class('FakeKernel')

        # Assert an error is raised when retrieving an unregistered peak
        assert 'is not registered' in str(info.value)


if __name__ == '__main__':
    pass

