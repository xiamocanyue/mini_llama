#include "include/cuda_quant.h"

#include <stdexcept>
#include <vector>

namespace mini_llama {
namespace {

std::runtime_error CudaQuantNotBuilt() {
    return std::runtime_error("CUDA quant kernels were not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine.");
}
} // namespace

bool CheckCudaQuantBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA
Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
    (void)x;
    (void)q8_0_device;
    (void)q8_0_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x,
                                    const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
    (void)x;
    (void)q8_0_device;
    (void)q8_0_block_count;
    (void)weight_shape;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
    (void)x;
    (void)q4_0_device;
    (void)q4_0_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
    (void)x;
    (void)q4_0_device;
    (void)q4_0_block_count;
    (void)weight_shape;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
    (void)x;
    (void)q4_1_device;
    (void)q4_1_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
    (void)x;
    (void)q4_1_device;
    (void)q4_1_block_count;
    (void)weight_shape;
    (void)device_id;
    throw CudaQuantNotBuilt();
}

#endif

}  