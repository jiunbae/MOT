#!/usr/bin/env bash

cd src/cuda
echo "Compiling psroi pooling kernels by nvcc..."
nvcc -c -o psroi_pooling.cu.o psroi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py