#pragma once 


#include "include/tensor.h"
#include "include/cuda_tensor.h"
#include "include/model.h"


namespace mini_llama {

bool CheckCudaOpsBuild();

//逐元素处理
Tensor CudaElementMutmul(const Tensor& a, const Tensor& b, int device_id = 0);
CudaTensor CudaElementMutmulDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id = 0);

Tensor CudaElementAdd(const Tensor& a, const Tensor& b, int device_id = 0);
CudaTensor CudaElementAddDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id = 0);
CudaTensor CudaElementAddDeviceWeight(const CudaTensor& a, const void* b_data, const std::vector<int>& b_shape, int device_id = 0);

Tensor CudaSilu(const Tensor& x, int device_id = 0);
CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id = 0);

Tensor CudaSoftmax(const Tensor& x, int device_id = 0);

/*
CudaRmsNorm
    输入和输出都是 CPU Tensor
    CPU -> GPU -> CPU
    适合兼容旧的 host 调用

CudaRmsNormDeviceInput
    输入 x 在 GPU
    weight 仍在 CPU
    每次把 weight 上传到 GPU

CudaRmsNormDeviceWeight
    输入和 weight 都在 GPU
    不需要重复拷贝 weight
    是主 CUDA 前向更适合使用的版本
*/
Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps, int device_id = 0);
CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight, float eps, int device_id = 0);
CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data, const std::vector<int>& weight_shape, float eps, int device_id = 0);

void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType type = RopeType::kNormal, int device_id = 0);
void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads, int head_dim, int pos, 
                float theta, RopeType type = RopeType::kNormal, int device_id = 0);

CudaTensor CudaEmbedding(const void* embedding_data, const std::vector<int>& embedding_shape, int token_id, int device_id = 0);


}