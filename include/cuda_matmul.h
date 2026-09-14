#pragma once

#include <vector>

#include "include/tensor.h"
#include "include/cuda_tensor.h"

namespace mini_llama {

bool CheckCudaMatmulBuilt();

// F32 matrix multiplication on CUDA using cuBLAS.
// a: [m, k], b: [k, n], result: [m, n]
Tensor CudaMatmul(const Tensor& a, const Tensor& b, int device_id = 0);

// F32 Linear projection on CUDA using cuBLAS.
// x: [in_features] or [batch, in_features]
// weight: [out_features, in_features]
// bias: optional [out_features], applied after the cuBLAS Matmul
// result: [out_features] for 1D input, [batch, out_features] for 2D input
Tensor CudaLinear(const Tensor& x, const Tensor& weight,
                  const Tensor* bias = nullptr, int device_id = 0);

// F32 Linear projection using a weight matrix that already lives in CUDA device
// memory. The input activation is copied host->device, and the result is copied
// device->host.
Tensor CudaLinearDeviceWeight(const Tensor& x, const void* w_device,
                              const std::vector<int>& w_shape,
                              const Tensor* bias = nullptr, int device_id = 0);

CudaTensor CudaLinearDeviceInput(const CudaTensor& x, const void* w_device,
                                 const std::vector<int>& w_shape,
                                 int device_id = 0);

}  

