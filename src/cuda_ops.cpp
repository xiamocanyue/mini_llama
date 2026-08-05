#include "include/cuda_ops.h"


namespace mini_llama {
namespace {

std::runtime_error CudaNotBuiltOps() {
    return std::runtime_error("CUDA is not built, check use NVIDIA or switch");
}

} //namespace


bool CheckCudaOpsBuild() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif 
}

#ifndef MINI_LLAMA_USE_CUDA

//逐元素处理
Tensor CudaElementMutmul(const Tensor& a, const Tensor& b, int device_id = 0) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaNotBuiltOps();
}

CudaTensor CudaElementMutmulDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id = 0) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaNotBuiltOps();
}

Tensor CudaElementAdd(const Tensor& a, const Tensor& b, int device_id = 0) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaNotBuiltOps();
}
CudaTensor CudaElementAddDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id = 0) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaNotBuiltOps();  
}

CudaTensor CudaElementAddDeviceWeight(const CudaTensor& a, const void* b_data, const std::vector<int>& b_shape, int device_id = 0) {
    (void)a;
    (void)b_data;
    (void)b_shape;
    (void)device_id;
    throw CudaNotBuiltOps();
}

Tensor CudaSilu(const Tensor& x, int device_id = 0) {
    (void)x;
    (void)device_id;
    throw CudaNotBuiltOps();
}

CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id = 0) {
    (void)x;
    (void)device_id;
    throw CudaNotBuiltOps();
}


Tensor CudaSoftmax(const Tensor& x, int device_id = 0) {
    (void)x;
    (void)device_id;
    throw CudaNotBuiltOps();
}

Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps, int device_id = 0) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw CudaNotBuiltOps();
}

CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight, float eps, int device_id = 0) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw CudaNotBuiltOps();
}

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data, const std::vector<int>& weight_shape, float eps, int device_id = 0) {
    (void)x;
    (void)weight_data;
    (void)weight_shape;
    (void)eps;
    (void)device_id;
    throw CudaNotBuiltOps();
}

void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType type = RopeType::kNormal, int device_id = 0) {
    (void)q;
    (void)k;
    (void)pos;
    (void)theta;
    (void)type;
    (void)device_id;
    throw CudaNotBuiltOps();
}

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads, int head_dim,
                int pos, float theta, RopeType type = RopeType::kNormal, int device_id = 0) {
    (void)q;
    (void)k;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)pos;
    (void)theta;
    (void)type;
    (void)device_id;
    throw CudaNotBuiltOps();
}

CudaTensor CudaEmbedding(const void* embedding_data, const std::vector<int>& embedding_shape, int token_id, int device_id = 0) {
    (void)embeding_data;
    (void)embedint_shap;
    (void)token_id;
    (void)device_id;
    throw CudaNotBuiltOps();
}


#endif


}