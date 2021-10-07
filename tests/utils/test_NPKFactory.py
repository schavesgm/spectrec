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

from spectrec.utils import NPKFactory

class FakeClass:
    pass

class FakePeak(Peak):
    pass

class FakeKernel(Kernel):
    pass

class FakeNetwork(Network):
    pass

@pytest.fixture(scope="session")
def factory():
    yield NPKFactory()

class TestNPKFactory:

    def test_factory_initialiser(self, factory):
        """ Assert that the class already registers some values. """

        # Assert that some items are already in the factory
        assert 'GaussianPeak' in factory.registered_names
        assert 'DeltaPeak'    in factory.registered_names
        assert 'NRQCDKernel'  in factory.registered_names
        assert 'UNet'         in factory.registered_names

        # Assert that the peaks are correctly registered
        assert issubclass(factory.retrieve_class('GaussianPeak'), Peak)
        assert issubclass(factory.retrieve_class('DeltaPeak'),    Peak)
        assert issubclass(factory.retrieve_class('NRQCDKernel'),  Kernel)
        assert issubclass(factory.retrieve_class('UNet'),         Network)

    def test_factory_register(self, factory):
        """ Test the registration of some classes. """

        # Try registering some classes
        factory.register_class('FakePeak',    FakePeak)
        factory.register_class('FakeKernel',  FakePeak)
        factory.register_class('FakeNetwork', FakePeak)
        assert 'FakePeak'    in factory.registered_names
        assert 'FakeKernel'  in factory.registered_names
        assert 'FakeNetwork' in factory.registered_names

        # Now, try registering a non-valid class
        with pytest.raises(AssertionError) as info:
            factory.register_class('FakeClass', FakeClass)

        # Assert an error is raised
        assert 'must be a subclass of [Peak, Kernel, Network]' in str(info.value)

    def test_factory_retrieve(self, factory):
        """ Test retrieving some classes """

        # Try retrieving a class that is not registered
        with pytest.raises(KeyError) as info:
            factory.retrieve_class('FakeClass')

        # Assert an error is raised
        assert 'is not registered' in str(info.value)

    def test_factory_retrieve_all_same_type(self, factory):
        """ Test the behaviour of retrieve_all_same_type. """

        # Get all peak, kernel and network classes
        peaks    = factory.retrieve_all_of_same_type('peak')
        kernels  = factory.retrieve_all_of_same_type('kernel')
        networks = factory.retrieve_all_of_same_type('network')

        # Assert that some items are inside each of them
        assert all(n in peaks    for n in ['GaussianPeak', 'DeltaPeak'])
        assert all(n in kernels  for n in ['NRQCDKernel'])
        assert all(n in networks for n in ['UNet'])

if __name__ == '__main__':
    pass

