"""Default blocks for UNet construction"""
import torch.nn as nn
import torch.nn.functional as F


class SplitBlock(nn.Module):
    def __init__(self, in_shape, out_shape, hw_shape, layer):
        super().__init__()
        in_feats, in_size = in_shape
        out_feats, out_size = out_shape
        hw_feats, _ = hw_shape

        assert out_feats == hw_feats
        assert 2 * out_size == in_size

        self.layers = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.ELU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.ELU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        hw = self.layers(x)
        return self.max_pool(hw), hw


class CenterBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.ELU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class MergeBlock(nn.Module):
    def __init__(self, in_shape, out_shape, hw_shape, layer):
        super().__init__()
        in_feats, in_size = in_shape
        out_feats, out_size = out_shape
        hw_feats, hw_size = hw_shape

        assert hw_size == 2 * in_size
        assert out_size == 2 * in_size

        self.deconv = nn.ConvTranspose2d(in_feats, hw_feats,
                                         kernel_size=2, stride=2)
        self.highway = nn.Sequential(
            nn.Conv2d(hw_feats, hw_feats, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(hw_feats, hw_feats, kernel_size=3, padding=1)
        )
        self.layers = nn.Sequential(
            nn.Conv2d(hw_feats, out_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.ELU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.ELU(inplace=True)
        )

    def forward(self, up, highway):
        up = self.deconv(up)
        hw = self.highway(highway)
        out = F.elu(up + hw, inplace=True)
        stuff = self.layers(out)
        return stuff
