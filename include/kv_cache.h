#pragma once 

#include "include/tensor.h"

namespace mini_llama {
//[n_layers, max_seq_len, n_kv_head, head_dim]
struct KVCache {
    KVCache() = default;
    KVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);

    void Write(int layer, int pos, const Tensor& k, const Tensor& v);

    const float* GetKeyPtr(int layer, int pos, int kv_head) const;
    const float* GetValuePtr(int layer, int pos, int kv_head) const;


    Tensor keys;
    Tensor values;
};

}