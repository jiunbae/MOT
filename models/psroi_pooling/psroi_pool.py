import torch.nn as nn

from .functional import PsRoIPool2DFunction


class PsRoIPool2D(nn.Module):
    def __init__(self, pooled_height: int, pooled_width: int, spatial_scale: float,
                 group_size: int = 0, output_dim: int = 1):
        super().__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

        self.group_size = int(group_size or self.pooled_width)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        return PsRoIPool2DFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size,
                                   self.output_dim)(features, rois)
