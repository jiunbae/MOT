import torch
import torch.nn as nn

from .functional import prroi_pool2d

__all__ = ['PrRoIPool2D']


class PrRoIPool2D(nn.Module):
    def __init__(self, pooled_height: int, pooled_width: int, spatial_scale: float):
        super().__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features: torch.Tensor, rois: torch.Tensor):
        return prroi_pool2d(features, rois,
                            self.pooled_height, self.pooled_width, self.spatial_scale)
