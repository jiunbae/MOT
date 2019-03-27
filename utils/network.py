import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.interpolate = F.interpolate

    def forward(self, x):
        return self.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                                mode=self.mode, align_corners=self.align_corners)


class DilationLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size: int = 3, dilation: int = 1):
        super(DilationLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2 * dilation), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConcatTable(nn.Module):
    def __init__(self, *args):
        super(ConcatTable, self).__init__()

        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def __getitem__(self, idx: int):
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, x: torch.Tensor):
        result = None

        for module in self._modules.values():
            out = module(x)
            result = out if result is None else result + out

        return result
