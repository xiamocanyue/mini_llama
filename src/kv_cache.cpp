#include "include/kv_cache.h"

#include <cstring>

namespace mini_llama {
namespace {

void Check4DShape(const Tensor& t, const char* caller) {
    if (t.DimNum() != 4) {
        throw std::runtime_error(std::string(caller) + "is not 4D tensor");
    }
}

void Check2DShape(const Tensor& t, const char* caller) {
    if (t.DimNum() != 2) {
        throw std::runtime_error(std::string(caller) + "is not 2D tensor");
    }
}
} // namespace

KVCache::KVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
    keys = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
    values = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}
//[n_layers, max_seq_len, x, x] 写入kv
void KVCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
    Check4DShape(keys, "keys");
    Check4DShape(values, "values");
    Check2DShape(k, "input_key");
    Check2DShape(v, "input_value");
    if (k.shape != v.shape) {
        throw std::runtime_error("KVCache : key, value expected [n_kv_heads, head_dim] tensor");
    }

    int n_kv_heads = k.shape[0];
    int head_dim = k.shape[1];
    if (layer < 0 || layer >= keys.shape[0]) {
        throw std::runtime_error("KVCache: write layer is not positive");
    }
    if (pos < 0 || pos >= keys.shape[1]) {
        throw std::runtime_error("KVCache: write pos is not positive");
    }
    if (n_kv_heads != keys.shape[2] || head_dim != keys.shape[3]) {
        throw std::runtime_error("KVCache: write tensor is not match shape");
    }

    //[layers, pos, x, x]
    size_t layer_stride = static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
    size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
    size_t kv_head_stride = static_cast<size_t>(keys.shape[3]);
    
    size_t start = layer * layer_stride + pos * pos_stride;
    for (int head = 0; head < n_kv_heads; head++) {
        size_t offset = start + head * kv_head_stride;
        std::memcpy(&keys.data[offset], &k.data[head * head_dim], sizeof(float) * head_dim);
        std::memcpy(&values.data[offset], &v.data[head * head_dim], sizeof(float) * head_dim);
    }
}

const float* KVCache::GetKeyPtr(int layer, int pos, int kv_head) const {
    Check4DShape(keys, "keys");
    if (layer < 0 || layer >= keys.shape[0]) {
        throw std::runtime_error("KVCache: GetKeyPtr layer is not positive");
    }
    if (pos < 0 || pos >= keys.shape[1]) {
        throw std::runtime_error("KVCache: GetKeyPtr pos is not positive");
    }
    if (kv_head < 0 || kv_head >= keys.shape[2]) {
        throw std::runtime_error("KVCache: GetKeyPtr kv_head is not positive");
    }

    size_t layer_stride = static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
    size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
    size_t kv_head_stride = static_cast<size_t>(keys.shape[3]);
    return &keys.data[layer * layer_stride + pos * pos_stride + kv_head * kv_head_stride];
}

const float* KVCache::GetValuePtr(int layer, int pos, int kv_head) const {
    Check4DShape(values, "values");
    if (layer < 0 || layer >= values.shape[0]) {
        throw std::runtime_error("KVCache: GetValuePtr layer is not positive");
    }
    if (pos < 0 || pos >= values.shape[1]) {
        throw std::runtime_error("KVCache: GetValuePtr pos is not positive");
    }
    if (kv_head < 0 || kv_head >= values.shape[2]) {
        throw std::runtime_error("KVCache: GetValuePtr kv_head is not positive");
    }

    size_t layer_stride = static_cast<size_t>(values.shape[1] * values.shape[2] * values.shape[3]);
    size_t pos_stride = static_cast<size_t>(values.shape[2] * values.shape[3]);
    size_t kv_head_stride = static_cast<size_t>(values.shape[3]);
    return &values.data[layer * layer_stride + pos * pos_stride + kv_head * kv_head_stride];
}
}