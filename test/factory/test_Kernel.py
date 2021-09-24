# Import some third party modules
import pytest
import torch

# Import some user-defined modules
from spectrec.factory import NRQCDKernel

@pytest.fixture(scope = "session")
def generate_kernel():
    yield NRQCDKernel(10, 1000, [8.0, 0.0])

class TestKernel:
    """ Class used to test the Kernel module. """

    def test_initialiser(self, generate_kernel):
        """ Test the constructor of the base abstract class. """
        assert generate_kernel.Nt == 10
        assert generate_kernel.Nw == 1000
        assert generate_kernel.w_range      == [0.0, 8.0]
        assert generate_kernel._kernel_type == 'NRQCDKernel'

    def test_sizes(self, generate_kernel):
        """ Test that the sizes of the generated data are correct. """
        assert generate_kernel.tau.shape    == torch.Size([generate_kernel.Nt])
        assert generate_kernel.omega.shape  == torch.Size([generate_kernel.Nw])
        assert generate_kernel.kernel.shape == torch.Size([generate_kernel.Nt, generate_kernel.Nw])

        # Assert the dw is correct
        wM, wm = max(generate_kernel.w_range), min(generate_kernel.w_range)
        assert generate_kernel.dw == (wM - wm) / generate_kernel.Nw

    def test_recalculate(self, generate_kernel):
        """ Test the data is correctly recalculated. """

        # Get the w_range
        w0, wf = generate_kernel.w_range

        # The data is the initial one
        assert torch.all(torch.eq(generate_kernel.tau,   torch.arange(0, generate_kernel.Nt)))
        assert torch.all(torch.eq(generate_kernel.omega, torch.linspace(w0, wf, generate_kernel.Nw)))

        # Change some contents in the kernel
        generate_kernel.Nt = 100
        generate_kernel.Nw = 1000

        # The data has been changed and the recalculation is correct
        assert torch.all(torch.eq(generate_kernel.tau,   torch.arange(0, generate_kernel.Nt)))
        assert torch.all(torch.eq(generate_kernel.omega, torch.linspace(w0, wf, generate_kernel.Nw)))

    def test_kernel_nrqcd(self, generate_kernel):
        """ Test the NRQCD kernel data. """

        # Get the tau and omega data of the kernel
        tau, omega = generate_kernel.tau, generate_kernel.omega

        # Get the Nt and Nw values from the kernel
        Nt, Nw = generate_kernel.Nt, generate_kernel.Nw

        # Assert the kernel is correct at these values
        assert torch.all(torch.eq(generate_kernel.kernel, (-tau.view(Nt, 1) * omega.view(1, Nw)).exp()))

if __name__ == '__main__':
    pass
