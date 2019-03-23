import os
import torch
from torch.utils.ffi import create_extension

if __name__ == '__main__':
    create_extension(
        '_ext.psroi_pooling',
        headers=['src/psroi_pooling_cuda.h'],
        sources=['src/psroi_pooling_cuda.c'],
        define_macros=[('WITH_CUDA', None)],
        relative_to=__file__,
        with_cuda=True,
        extra_objects=[os.path.join(os.path.dirname(os.path.realpath(__file__)), fname) for fname in [
            'src/cuda/psroi_pooling.cu.o'
        ]]
    ).build()
