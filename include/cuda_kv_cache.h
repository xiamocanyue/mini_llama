#pragma once 

#include "include/kv_cache.h"
#include "include/cuda_data.h"
#include "include/cuda_tensor.h"

namespace mini_llama {

bool CheckCudaKVCacheBuilt();

class CudaKVCache {
public:
	CudaKVCache() = default;
	CudaKVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id = 0);

	CudaKVCache(const CudaKVCache&) = delete;
	CudaKVCache& operator=(const CudaKVCache&) = delete;
	CudaKVCache(CudaKVCache&&) noexcept = default;
	CudaKVCache& operator=(CudaKVCache&&) noexcept = default;

	void Reset();
	void Reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id = 0);
	void Clear();

	void Write(int layer, int pos, const Tensor& k, const Tensor& v);
	void WriteDevice(int layer, int pos, const CudaTensor& k, const CudaTensor& v);

	Tensor ReadKey(int layer, int pos) const;
	Tensor ReadValue(int layer, int pos) const;
	Tensor ReadKeyHead(int layer, int pos, int kv_head) const;
	Tensor ReadValueHead(int layer, int pos, int kv_head) const;

	bool Empty() const {
		return keys_.Empty() || values_.Empty();
	}

	int GetNLayers() const {
		return n_layers_;
	}

	int GetMaxSeqLen() const {
		return max_seq_len_;
	}

	int GetNKVHeads() const {
		return n_kv_heads_;
	}

	int GetHeadDim() const {
		return head_dim_;
	}

	int GetDeviceId() const {
		return device_id_;
	}

	size_t GetBytes() const {
		return keys_.GetBytes() + values_.GetBytes();
	}

	const void* GetKeyData() const {
		if (keys_.Empty()) {
			throw std::runtime_error("Cuda kv_cache: key cache is empty");
		}
		return keys_.GetData();
	}

	const void* GetValueData() const {
		if (values_.Empty()) {
			throw std::runtime_error("Cuda kv_cache: values cache is empty");
		}
		return values_.GetData();
	}

private:
	void ValidateIndices(int layer, int pos) const;
	void ValidateHeadIndex(int kv_head) const;
	void ValidateWriteTensors(const Tensor& k, const Tensor& v) const;
	size_t SlotOffsetBytes(int layer, int pos) const;
	size_t HeadOffsetBytes(int layer, int pos, int kv_head) const;

private:
	CudaTensor keys_;
	CudaTensor values_;
	int n_layers_{0};
	int max_seq_len_{0};
	int n_kv_heads_{0};
	int head_dim_{0};
	int device_id_{0};
};

}
