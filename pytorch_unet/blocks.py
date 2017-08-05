"""Default blocks for UNet construction"""
import torch.nn as nn
import torch.nn.functional as F


class SplitBlock(nn.Module):
    def __init__(self, in_shape, up_shape, hor_shape, layer):
        super().__init__()
        in_feats, in_size = in_shape
        up_feats, up_size = up_shape
        hor_feats, _ = hor_shape

        assert up_feats == hor_feats
        assert 2 * up_size == in_size

        self.layers = nn.Sequential(
            nn.Conv2d(in_feats, up_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(up_feats),
            nn.ELU(inplace=True),
            nn.Conv2d(up_feats, up_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(up_feats),
            nn.ELU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        hor = self.layers(x)
        return self.max_pool(hor), hor


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
    def __init__(self, in_shape, out_shape, hor_shape, layer):
        super().__init__()
        in_feats, in_size = in_shape
        out_feats, out_size = out_shape
        hor_feats, hor_size = hor_shape

        assert hor_size == 2 * in_size
        assert out_size == 2 * in_size

        self.deconv = nn.ConvTranspose2d(in_feats, hor_feats,
                                         kernel_size=2, stride=2)
        self.highway = nn.Sequential(
            nn.Conv2d(hor_feats, hor_feats, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(hor_feats, hor_feats, kernel_size=3, padding=1)
        )
        self.layers = nn.Sequential(
            nn.Conv2d(hor_feats, out_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.ELU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.ELU(inplace=True)
        )

    def forward(self, up, hor):
        up = self.deconv(up)
        hor = self.highway(hor)
        out = F.elu(up + hor, inplace=True)
        stuff = self.layers(out)
        return stuff


class FinalBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.layer = nn.Conv2d(in_feats, out_feats, kernel_size=1)

    def forward(self, x):
        return self.layer(x)
