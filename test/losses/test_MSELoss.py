# Load some third-party modules
import pytest
import torch

# Load some user-defined modules
from spectrec.losses import MSELoss

class TestMSELoss:
    """ Class used to test the MSELoss class. """

    def test_behaviour(self):
        """ Test the behaviour of the MSELoss class. """

        # Generate a MSELoss object to test it
        mseloss = MSELoss()

        # Some tensors to calculate the forward pass
        input_1 = torch.arange(0, 100, dtype = torch.float32).view(1, 1, 100)

        # Generate several losses
        loss_1 = mseloss(input_1, input_1)
        loss_2 = mseloss(input_1, 2 * input_1)

        # Assert some equalities on the data
        assert float(loss_1) == pytest.approx(0.0)
        assert float(loss_2) == float((input_1 ** 2).sum())

        # Now, test the behaviour of the loss function using several batches
        input_2 = torch.cat([input_1 for i in range(1000)], dim = 0)

        # Generate several losses
        loss_3 = mseloss(input_2, input_2)
        loss_4 = mseloss(input_2, 2 * input_2)

        # Assert some equalities
        assert float(loss_3) == pytest.approx(0.0)
        assert float(loss_4) == pytest.approx(float((input_1 ** 2).sum()))

if __name__ == '__main__':
    pass

