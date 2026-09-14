#pragma once 

#include "include/cuda_tensor.h"
#include "include/cuda_kv_cache.h"

namespace mini_llama {

bool CheckCudaAttentionBuilt();

Tensor CudaAttentionDecode(const Tensor& q, const CudaKVCache& kv_cache, int layer, 
					int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim, int device_id = 0);

CudaTensor CudaAttentionDecodeDeviceInputs(const CudaTensor& q, const CudaKVCache& kv_cache, int layer, 
					int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim, int device_id = 0);

}