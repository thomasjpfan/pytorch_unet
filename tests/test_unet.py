import pytest
import torch
from torch.autograd import Variable

from pytorch_unet import UNet


def test_invalid_shape():
    with pytest.raises(ValueError):
        UNet(input_shape=(3, 33), layers=3, num_classes=1)


def test_unet_double_center():
    net = UNet(input_shape=(3, 32), layers=3, num_classes=2, double_center_features=True)
    x = torch.randn(10, 3, 32, 32)
    x_var = Variable(x)

    output = net(x_var)
    output_size = output.size()

    assert output_size[0] == 10
    assert output_size[1] == 2
    assert output_size[2] == 32
    assert output_size[3] == 32


def test_unet_no_double_center():
    net = UNet(input_shape=(3, 32), layers=3, num_classes=2, double_center_features=False)
    x = torch.randn(10, 3, 32, 32)
    x_var = Variable(x)

    output = net(x_var)
    output_size = output.size()

    assert output_size[0] == 10
    assert output_size[1] == 2
    assert output_size[2] == 32
    assert output_size[3] == 32
