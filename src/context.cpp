#include "include/context.h"

#include <cmath>

namespace mini_llama {


MiniLlamaContext::MiniLlamaContext(const MiniLlamaModel* model_ptr) : model(model_ptr) {
    if (model != nullptr) {
        const auto& c = model->config;
        kv_cache = KVCache(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim);
        if (c.head_dim <= 0 || c.head_dim % 2 != 0) {
            throw std::runtime_error("MiniLlamaContext: head_dim must be positive and even");
        }

        int half_dim = c.head_dim / 2;
        rope_cos.resize(static_cast<size_t>(c.max_seq_len * half_dim));
        rope_sin.resize(static_cast<size_t>(c.max_seq_len * half_dim));
        for (int pos = 0; pos < c.max_seq_len; ++pos) {
            for (int i = 0; i < half_dim; ++i) {
                float freq = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * i) / static_cast<float>(c.head_dim));
                rope_cos[pos * half_dim + i] = std::cos(pos * freq);
                rope_sin[pos * half_dim + i] = std::sin(pos * freq);
            }
        }

        attention_scores.resize(static_cast<size_t>(c.n_heads * c.max_seq_len));
        attention_probs.resize(static_cast<size_t>(c.n_heads * c.max_seq_len));

    }
}
}
