# Import some built-in modules
import os
import shutil

# Import some third-party modules
import torch
import pytest

# Load some user-defined modules
from spectrec.network import UNet


@pytest.fixture(scope="session")
def output_path():
    return './status/network'


class TestNetwork:
    """ Test the Network base class using UNet as interface. """

    def test_setparams_cpu(self):
        """ Test the correctness of the parameter copy. """

        # Generate two different networks
        net1, net2 = UNet(10, 100), UNet(10, 100)

        # Assert the dictionary keys are the same -- They should
        assert net1.state_dict().keys() == net2.state_dict().keys()

        # Copy the parameters of the second network onto the first one
        net1.set_params(net2.state_dict())

        # Check that the parameters are the same, so they are copied
        for w1, w2 in zip(net1.state_dict().values(), net2.state_dict().values()):
            assert torch.all(torch.eq(w1, w2))

    @pytest.mark.skipif('not torch.cuda.is_available()', reason='CUDA is not available')
    def test_setparams_gpu(self):
        """ Set the parameters using a GPU dataparallel module. """

        # Generate a network
        net1 = UNet(10, 100)

        # Generate a parallelised network
        net2 = torch.nn.DataParallel(UNet(10, 100))

        # Copy the parameters of the parallelised network onto the first one
        net1.set_params(net2.state_dict(), 'module.')

        # Check that the parameters are the same, so they are copied
        for w1, w2 in zip(net1.state_dict().values(), net2.state_dict().values()):
            assert torch.all(torch.eq(w1, w2.cpu()))

    def test_IO(self, output_path):
        """ Test the networks ability to save/load parameters. """

        # Generate two neural networks
        net_1, net_2 = UNet(64, 20), UNet(100, 20)

        # Change the name of the second neural network
        net_2.name = 'test'

        # Save the parameters of the network into a file
        net_1.save_params(path=output_path)
        net_2.save_params(path=output_path)

        # Assert both parameters exist
        assert os.path.exists(f'{output_path}/UNet/test.pt')
        assert os.path.exists(f'{output_path}/UNet/noName.pt')

        # Assert that we can load the parameters from the file
        net_1.load_params(path=output_path, verbose=False)
        net_2.load_params(path=output_path, verbose=False)

        # Delete the newly created folder
        shutil.rmtree('./status')


if __name__ == '__main__':
    pass
