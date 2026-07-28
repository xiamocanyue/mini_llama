#pragma once 


#include "include/tensor.h"
#include "include/quantized_tensor.h"

namespace mini_llama {

enum class RopeType {
    kNormal,
    kNeoX,
};

struct ModelConfig {
    int vocab_size = 128;
    int dim = 32;
    int hidden_dim = 86;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 4;
    int head_dim = 8;
    int max_seq_len = 128;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    RopeType rope_type = RopeType::kNormal;
};

struct LayerWeights {
    QuantizedTensor wq;      // [n_heads * head_dim, dim]  F32/Q8_0/Q4_0
    QuantizedTensor wk;      // [n_kv_heads * head_dim, dim]
    QuantizedTensor wv;      // [n_kv_heads * head_dim, dim]
    Tensor bq;               // optional [n_heads * head_dim] F32
    Tensor bk;               // optional [n_kv_heads * head_dim] F32
    Tensor bv;               // optional [n_kv_heads * head_dim] F32
    QuantizedTensor wo;      // [dim, n_heads * head_dim]
    Tensor attention_norm;   // [dim] F32
    Tensor ffn_norm;         // [dim] F32
    QuantizedTensor w_gate;  // [hidden_dim, dim]
    QuantizedTensor w_up;    // [hidden_dim, dim]
    QuantizedTensor w_down;  // [dim, hidden_dim]
};

struct MiniLlamaModel {
    ModelConfig config;
    bool is_loaded = false;
    std::string load_error;
    Tensor token_embedding;            // [vocab_size, dim] F32
    std::vector<LayerWeights> layers;  // [n_layers]
    Tensor final_norm;                 // [dim] F32
    QuantizedTensor lm_head;           // [vocab_size, dim] F32/Q8_0/Q4_0
    //std::shared_ptr<CudaModelWeights> cuda_weights;
};


size_t ModelWeightBytes(const MiniLlamaModel& model);

size_t ModelWeightBytesF32(const MiniLlamaModel& model);


}