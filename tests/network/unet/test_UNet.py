# Import some built-in modules
import os
import shutil

# Import some third-party modules
import torch
import pytest

# Load some user-defined modules
from spectrec.network import UNet

class TestUNet:


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

    def test_IO(self):
        """ Test the networks ability to save/load parameters. """

        # Generate a neural network object
        net = UNet(64, 20)

        # Save the parameters of the network into a file
        net.save_params('test')

        # Assert the file exists
        assert os.path.exists('./status/net/test.pt')

        # Assert that we can load the parameters from the file
        net.load_params('test', verbose=False)

        # Delete the newly created folder
        shutil.rmtree('./status')


if __name__ == '__main__':
    pass