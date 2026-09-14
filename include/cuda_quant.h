#pragma once

#include <vector>

#include "include/cuda_tensor.h"
#include "include/quantized_tensor.h"
#include "include/tensor.h"

namespace mini_llama {

bool CheckCudaQuantBuilt();
    
Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias, int device_id);

Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_weight, size_t q8_0_block_count, 
							const std::vector<int>& weight_shape, const Tensor* bias, int device_id);

CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x, const void* q8_0_weight, size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape, int device_id);

Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias, int device_id);

Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_weight, size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape, const Tensor* bias, int device_id);

CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x, const void* q4_0_weight, size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape, int device_id);

Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias, int device_id);

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_weight, size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape, const Tensor* bias, int device_id);

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x, const void* q4_1_weight, size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape, int device_id = 0);

}

