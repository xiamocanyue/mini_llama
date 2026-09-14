#include "include/cuda_attention.h"

namespace mini_llama {
namespace {
std::runtime_error CudaAttentionNotBuilt() {
	return std::runtime_error("Cuda attention is not built, reconfigure on matchine");
}
}  //namespace

bool CheckCudaAttentionBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
	return true;
#else
	return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA
Tensor CudaAttentionDecode(const Tensor& q, const CudaKVCache& kv_cache, int layer,
						int pos, int max_seq_len, int n_heads, int n_kv_heads,
						int head_dim, int device_id) {
	(void)q;
	(void)kv_cache;
	(void)layer;
	(void)pos;
	(void)max_seq_len;
	(void)n_heads;
	(void)n_kv_heads;
	(void)head_dim;
	(void)device_id;
	throw CudaAttentionNotBuilt();
}

CudaTensor CudaAttentionDecodeDeviceInputs(const CudaTensor& q,
						const CudaKVCache& kv_cache, int layer, int pos,
						int max_seq_len, int n_heads, int n_kv_heads,
						int head_dim, int device_id) {
	(void)q;
	(void)kv_cache;
	(void)layer;
	(void)pos;
	(void)max_seq_len;
	(void)n_heads;
	(void)n_kv_heads;
	(void)head_dim;
	(void)device_id;
	throw CudaAttentionNotBuilt();
}
#endif






}
