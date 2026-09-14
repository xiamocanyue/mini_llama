#include "include/cuda_kv_cache.h"

#include <limits> 

namespace mini_llama {
namespace {

std::runtime_error CudaKVCacheNotBuilt() {
	return std::runtime_error("Cuda kv_cache is not build, reconfigure Cuda is open");
}

void CheckPositive(int value, const char* caller) {
	if (value <= 0) {
		throw std::runtime_error("Cuda kv_cache: " + std::string(caller) + "must be positive");
	}
}

size_t GetAndCheckMul(size_t a, size_t b, const char* caller) {
	if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
		throw std::runtime_error("Cuda kv_cache: size overflow for " + std::string(caller));
	}
	return a * b;
}

} //namespace

bool CheckCudaKVCacheBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
	return true;
#else
	return false;
#endif
}

CudaKVCache::CudaKVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id) {
	Reset(n_layers, max_seq_len, n_kv_heads, head_dim, device_id);
}

void CudaKVCache::Clear() {
#ifdef MINI_LLAMA_USE_CUDA
	if (Empty()) {
		return;
	}
	CudaSetDevice(device_id_);
	keys_.Upload(Tensor(keys_.Shape(), 0.0f));
	values_.Upload(Tensor(values_.Shape(), 0.0f));

#else
	throw CudaKVCacheNotBuilt();
#endif
}

void CudaKVCache::Reset() {
	keys_.Reset();
	values_.Reset();
	n_layers_ = 0;
	max_seq_len_ = 0;
	n_kv_heads_ = 0;
	head_dim_ = 0;
	device_id_ = 0;
}

void CudaKVCache::Reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
	CheckPositive(n_layers, "n_layers");
	CheckPositive(max_seq_len, "max_seq_len");
	CheckPositive(n_kv_heads, "n_kv_heads");
	CheckPositive(head_dim, "head_dim");

	n_layers_ = n_layers;
	max_seq_len_ = max_seq_len;
	n_kv_heads_ = n_kv_heads;
	head_dim_ = head_dim;
	device_id_ = device_id;
	keys_.Reset({n_layers, max_seq_len, n_kv_heads, head_dim}, device_id);
	values_.Reset({n_layers, max_seq_len, n_kv_heads, head_dim}, device_id);
	Clear();
#else
	(void)n_layers;
	(void)max_seq_len;
	(void)n_kv_heads;
	(void)head_dim;
	(void)device_id;
	throw CudaKVCacheNotBuilt();
#endif
}

void CudaKVCache::ValidateIndices(int layer, int pos) const {
	if (Empty()) {
		throw std::runtime_error("CudaKvCache: cache is empty");
	}
	if (layer < 0 || layer >= n_layers_) {
		throw std::out_of_range("CudaKvCache layer out of range");
	}
	if (pos < 0 || pos >= max_seq_len_) {
		throw std::out_of_range("CudaKvCache position out of range");
	} 
}

void CudaKVCache::ValidateHeadIndex(int kv_head) const{
	if (kv_head < 0 || kv_head >= n_kv_heads_) {
		throw std::out_of_range("CudaKvCache head out of range");
	}
}

void CudaKVCache::ValidateWriteTensors(const Tensor& k, const Tensor& v) const {
	if (k.DimNum() != 2 || v.DimNum() != 2 || k.shape != v.shape) {
		throw std::out_of_range("CudaKvCache::Write expected matching [n_kv_heads, head_dim] tensors");
	}
	if (k.shape[0] != n_kv_heads_ || k.shape[1] != head_dim_) {
		throw std::out_of_range("CudaKvCache::Write tensor shape does not match cache shape");
	}	
}

size_t CudaKVCache::SlotOffsetBytes(int layer, int pos) const {
	size_t offset = GetAndCheckMul(static_cast<size_t>(layer), static_cast<size_t>(max_seq_len_), "layer * max_seq_len");
	if (offset > std::numeric_limits<size_t>::max() - pos) {
		throw std::runtime_error("Cuda kv_cache: size overflow for offset + pos");
	}
	offset += static_cast<size_t>(pos);
	offset = GetAndCheckMul(offset, static_cast<size_t>(n_kv_heads_), "(layer * max_seq + pos) * n_kv_heads");
	offset = GetAndCheckMul(offset, static_cast<size_t>(head_dim_), "(layer * max_seq + pos) * n_kv_heads * head_dim");
	return GetAndCheckMul(offset, sizeof(float), "offset * sizeof(float)");
}

size_t CudaKVCache::HeadOffsetBytes(int layer, int pos, int kv_head) const {
	size_t offset_base = SlotOffsetBytes(layer, pos);
	size_t offset_new = GetAndCheckMul(static_cast<size_t>(kv_head), static_cast<size_t>(head_dim_),  "kv_head * head_dim");
	offset_new = GetAndCheckMul(offset_new, sizeof(float), "offset_new * sizeof(float)");
	if (offset_new > std::numeric_limits<size_t>::max() - offset_base) {
		throw std::runtime_error("Cuda kv_cache: size overflow for offset + offset_new");
	}
	return offset_base + offset_new;
}

void CudaKVCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
#ifdef MINI_LLAMA_USE_CUDA
	ValidateIndices(layer, pos);
	ValidateWriteTensors(k, v);
	size_t offset = SlotOffsetBytes(layer, pos);
	size_t bytes_to_copy = k.Size() * sizeof(float);
	char* key_dst = static_cast<char*>(keys_.GetData()) + offset;
	char* value_dst = static_cast<char*>(values_.GetData()) + offset;
	
	CudaSetDevice(device_id_);
	CudaMemcpyBytes(key_dst, k.data.data(), bytes_to_copy, MemcpyKind::kHostToDevice);
	CudaMemcpyBytes(value_dst, v.data.data(), bytes_to_copy, MemcpyKind::kHostToDevice);
#else
	(void)layer;
	(void)pos;
	(void)k;
	(void)v;
	throw CudaKVCacheNotBuilt();
#endif	
}

void CudaKVCache::WriteDevice(int layer, int pos, const CudaTensor& k, const CudaTensor& v) {
#ifdef MINI_LLAMA_USE_CUDA
	ValidateIndices(layer, pos);
	if (keys_.GetDeviceId() != device_id_ || values_.GetDeviceId() != device_id_) {
		throw std::runtime_error("Cuda kv_cache: tensor is not on same device");
	}
	const size_t expected = GetAndCheckMul(static_cast<size_t>(n_kv_heads_), static_cast<size_t>(head_dim_), "n_kv_heads * head_dim");
	if (k.GetSize() != expected || v.GetSize() != expected) {
		throw std::out_of_range("Cuda kv_cache: tensor shape is not match");
	}

	size_t offset = SlotOffsetBytes(layer, pos);
	char* key_dst = static_cast<char*>(keys_.GetData()) + offset;
	char* value_dst = static_cast<char*>(values_.GetData()) + offset;

	CudaSetDevice(device_id_);
	CudaMemcpyBytes(key_dst, k.GetData(), expected * sizeof(float), MemcpyKind::kDeviceToDevice);
	CudaMemcpyBytes(value_dst, v.GetData(), expected * sizeof(float), MemcpyKind::kDeviceToDevice);
#else
	(void)layer;
	(void)pos;
	(void)k;
	(void)v;
	throw CudaKvCacheNotBuilt();
#endif
}

Tensor CudaKVCache::ReadKey(int layer, int pos) const {
#ifdef MINI_LLAMA_USE_CUDA
	ValidateIndices(layer, pos);
	size_t offset = SlotOffsetBytes(layer, pos);
	const char* src = static_cast<const char*>(keys_.GetData()) + offset;
	Tensor dst({n_kv_heads_, head_dim_}, 0.0f);

	CudaSetDevice(device_id_);
	CudaMemcpyBytes(dst.data.data(), src, dst.Size() * sizeof(float), MemcpyKind::kDeviceToHost);
	return dst;

#else
	(void)layer;
	(void)pos;
	throw CudaKVCacheNotBuilt();
#endif
}

Tensor CudaKVCache::ReadValue(int layer, int pos) const {
#ifdef MINI_LLAMA_USE_CUDA
	ValidateIndices(layer, pos);
	size_t offset = SlotOffsetBytes(layer, pos);
	const char* src = static_cast<const char*>(values_.GetData()) + offset;
	Tensor dst({n_kv_heads_, head_dim_}, 0.0f);

	CudaSetDevice(device_id_);
	CudaMemcpyBytes(dst.data.data(), src, dst.Size() * sizeof(float), MemcpyKind::kDeviceToHost);
	return dst;
#else
	(void)layer;
	(void)pos;
	throw CudaKVCacheNotBuilt();
#endif
}

Tensor CudaKVCache::ReadKeyHead(int layer, int pos, int kv_head) const {
#ifdef MINI_LLAMA_USE_CUDA
	ValidateIndices(layer, pos);
	ValidateHeadIndex(kv_head);

	size_t offset = HeadOffsetBytes(layer, pos, kv_head);
	const char* src = static_cast<const char*>(keys_.GetData()) + offset;
	Tensor dst({head_dim_}, 0.0f);

	CudaSetDevice(device_id_);
	CudaMemcpyBytes(dst.data.data(), src, dst.Size() * sizeof(float), MemcpyKind::kDeviceToHost);
	return dst;

#else
	(void)layer;
	(void)pos;
	(void)kv_head;
	throw CudaKVCacheNotBuilt();
#endif
}

Tensor CudaKVCache::ReadValueHead(int layer, int pos, int kv_head) const {
#ifdef MINI_LLAMA_USE_CUDA
	ValidateIndices(layer, pos);
	ValidateHeadIndex(kv_head);

	size_t offset = HeadOffsetBytes(layer, pos, kv_head);
	const char* src = static_cast<const char*>(values_.GetData()) + offset;
	Tensor dst({head_dim_}, 0.0f);

	CudaSetDevice(device_id_);
	CudaMemcpyBytes(dst.data.data(), src, dst.Size() * sizeof(float), MemcpyKind::kDeviceToHost);
	return dst;
#else
	(void)layer;
	(void)pos;
	(void)kv_head;
	throw CudaKVCacheNotBuilt();
#endif
}

}