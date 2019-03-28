import torch


try:
    from os.path import join as pjoin, dirname
    from torch.utils.cpp_extension import load as load_extension
    root_dir = pjoin(dirname(__file__), 'src')
    _psroi_pooling = load_extension(
        '_psroi_pooling', [
            pjoin(root_dir, 'psroi_pooling_cuda.c'),
            pjoin(root_dir, 'cuda/psroi_pooling_kernel.cu')
        ], verbose=True,
    )
except ImportError:
    raise ImportError('Can not compile Position Sensitive RoI Pooling library.')


class PsRoIPool2DFunction(torch.autograd.Function):
    def __init__(self, pooled_height: int, pooled_width: int, spatial_scale: float,
                 group_size: int, output_dim: int):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

        self.output = None
        self.mappingchannel = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        num_rois = rois.size(0)

        output = features.new().resize_(
            num_rois, self.output_dim, self.pooled_height, self.pooled_width
        ).zero_()
        mappingchannel = torch.IntTensor(
            num_rois, self.output_dim, self.pooled_height, self.pooled_width
        ).zero_().cuda(features.get_device())

        rtn = _psroi_pooling.psroi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                        self.group_size, self.output_dim,
                                                        features, rois, output, mappingchannel)
        assert rtn > 0
        self.output = output
        self.mappingchannel = mappingchannel
        self.rois = rois
        self.feature_size = features.size()

        return output

    def backward(self, grad_output):
        assert (self.feature_size is not None and grad_output.is_cuda)

        grad_input = torch.zeros(*self.feature_size).cuda()

        _psroi_pooling.psroi_pooling_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                   self.output_dim, grad_output,
                                                   self.rois, grad_input, self.mappingchannel)
        return grad_input, None
