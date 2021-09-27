# Import some built-in modules
import os
import shutil

# Import some third-party modules
import torch
import pytest

# Load some user-defined modules
from spectrec.network import UNet

@pytest.fixture(scope="session")
def x_tensor():
    x = torch.arange(0, 64, dtype=torch.float32).view(1, 1, 64)
    yield torch.cat([x for _ in range(100)], dim=0)

class TestUNet:
    """ Test the UNet class. """

    def test_forward_pass(self, x_tensor):
        """ Test the forward pass of the network. """

        # Pass the tensor through the network
        output = UNet(x_tensor.shape[2], 32)(x_tensor)

        # Assert the sizes are correct
        assert output.ndim     == 3
        assert output.shape[0] == 100
        assert output.shape[1] == 1
        assert output.shape[2] == 32

    def test_preamplify(self):
        """ Test the preamplify step depending on network input. """

        # Generate two different networks
        unet1, unet2 = UNet(100, 32), UNet(572, 32)

        # Assert preamplify is in the state dictionary
        assert any('pre_amplify' in name[0] for name in unet1.named_modules())
        assert all('pre_amplify' not in name[0] for name in unet2.named_modules())


if __name__ == '__main__':
    pass
