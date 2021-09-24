# Import some third-party modules
import pytest

# Import some user-defined modules
from spectrec.factory import GaussianPeak
from spectrec.factory import DeltaPeak

@pytest.fixture(scope = "session")
def generate_limits():
    yield {'M': [0, 1], 'A': [1.0, 2.0], 'W': [5.0, 2.0], 'Ad': [10, 20], 'Md': [50, 70]}

@pytest.fixture(scope = "session")
def generate_gaussian(generate_limits):
    yield GaussianPeak(generate_limits, None)

@pytest.fixture(scope = "session")
def generate_delta(generate_limits):
    yield DeltaPeak(generate_limits, 'd')

class TestPeaks:
    ''' Class used to test the Peak module. '''

    def test_initialiser(self, generate_gaussian, generate_delta, generate_limits):
        """ Select some correct parameters from a dictionary of parameters. """

        # Get a reference to the gaussian and delta peaks
        gauss, delta = generate_gaussian, generate_delta

        # Assert some conditions on each of the peaks
        assert gauss.limits['M'] == sorted(generate_limits['M'])
        assert gauss.limits['A'] == sorted(generate_limits['A'])
        assert gauss.limits['W'] == sorted(generate_limits['W'])
        assert delta.limits['M'] == sorted(generate_limits['Md'])
        assert delta.limits['A'] == sorted(generate_limits['Ad'])

        # Assert the names are correct
        assert gauss._peak_type == 'GaussianPeak'
        assert delta._peak_type == 'DeltaPeak'

    def test_change_parameter(self, generate_gaussian, generate_delta):
        """ Test the change of parameters. """

        # Get a reference to the gaussian and delta peaks
        gauss, delta = generate_gaussian, generate_delta

        # Assert that we can change the parameter limits correctly
        gauss.set_parameter_limit('A', [100, 200])
        delta.set_parameter_limit('M', [1000, 5000])

        assert gauss.limits['A'] == sorted([100,  200])
        assert delta.limits['M'] == sorted([1000, 5000])

    def test_recalculate(self, generate_gaussian):
        """ Test that the recalculation is correct and yields correct results. """

        # Get a reference to the gaussian and delta peaks
        gauss = generate_gaussian
        
        # Test several times as randomness is present
        for t in range(10000):

            # Generate a new set of parameters
            gauss.recalculate();

            assert gauss.limits['M'][0] <= gauss.M < gauss.limits['M'][1]
            assert gauss.limits['A'][0] <= gauss.A < gauss.limits['A'][1]
            assert gauss.limits['W'][0] <= gauss.W < gauss.limits['W'][1]

if __name__ == '__main__':
    pass
