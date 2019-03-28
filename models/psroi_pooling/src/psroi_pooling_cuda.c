#include <math.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>

#include "cuda/psroi_pooling_kernel.h"


int psroi_pooling_forward_cuda(
    int pooled_height, int pooled_width,
    float spatial_scale, int group_size, int output_dim,
    at::Tensor& features, at::Tensor& rois,
    at::Tensor& output, at::Tensor& mappingchannel
) {
    int num_rois = rois.size(0);
	int size_rois = rois.size(1);

	int batch_size = features.size(0);
	int num_channels = features.size(1);
	int data_height = features.size(2);
	int data_width = features.size(3);


	if (size_rois!=5) {
		return -1;
	}

	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	PSROIPoolForwardLauncher(
	    features.data<float>(), spatial_scale, num_rois,
        data_height, data_width, num_channels,
        pooled_height, pooled_width, rois.data<float>(), group_size,
        output_dim, output.data<float>(), mappingchannel.data<int>(), stream
    );

	return 1;
}


int psroi_pooling_backward_cuda(
    int pooled_height, int pooled_width, float spatial_scale, int output_dim,
    at::Tensor& top_grad, at::Tensor& rois,
    at::Tensor& bottom_grad, at::Tensor& mappingchannel
) {
    int num_rois = rois.size(0);
    int size_rois =rois.size(1);

    int batch_size = bottom_grad.size(0);
    int num_channels = bottom_grad.size(1);
    int data_height = bottom_grad.size(2);
    int data_width = bottom_grad.size(3);

    if (size_rois != 5) {
        return -1;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    PSROIPoolBackwardLauncher(
        top_grad.data<float>(), mappingchannel.data<int>(), batch_size, num_rois,
        spatial_scale, num_channels, data_height, data_width,
        pooled_width, pooled_height, output_dim, bottom_grad.data<float>(), rois.data<float>(), stream
    );
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("psroi_pooling_forward_cuda", &psroi_pooling_forward_cuda, "PSRoIPooling_forward");
    m.def("psroi_pooling_backward_cuda", &psroi_pooling_backward_cuda, "PSRoIPooling_backward");
}
