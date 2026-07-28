#include "include/gguf_loader.h"

#include <sstream>
#include <fstream>
#include <limits>
#include <cstring>
#include <iostream>
#include <map>

#include "include/gguf.h"
#include "include/quantized_tensor.h"
#include "include/quant.h"


namespace mini_llama {
namespace {

/*
token_embd.weight       -> token_embedding
output_norm.weight      -> final_norm
output.weight           -> lm_head
blk.0.attn_q.weight     -> layers.0.wq
blk.0.attn_p.bias       -> layers.0.bp
blk.0.ffn_down.weight   -> layers.0.w_down
*/
std::string MapTensorName(const std::string& gguf_name) {
    if (gguf_name == "token_embd.weight") {
        return "token_embedding";
    }
    if (gguf_name == "output_norm.weight") {
        return "final_norm";
    }
    if (gguf_name == "output.weight") {
        return "lm_head";
    }

    if (gguf_name.rfind("blk.", 0) == 0) {
        size_t dot1 = gguf_name.find('.', 4);
        if (dot1 == std::string::npos) {
            return "";
        }
        std::string layer_str = gguf_name.substr(4, dot1 - 4);

        size_t dot2 = gguf_name.find('.', dot1 + 1);
        if (dot2 == std::string::npos) {
            return "";
        }
        std::string op = gguf_name.substr(dot1 + 1, dot2 - dot1 - 1);
        std::string kind = gguf_name.substr(dot2 + 1);
        std::string mapped_op;
        if (kind == "bias") {
            if (op == "attn_q") {
                mapped_op = "bq";
            } else if (op == "attn_k") {
                mapped_op = "bk";
            } else if (op == "attn_v") {
                mapped_op = "bv";
            } else {
                return "";
            }
        } else if (kind != "weight") {
            return "";
        } else if (op == "attn_norm") {
            mapped_op = "attention_norm";
        } else if (op == "attn_q") {
            mapped_op = "wq";
        } else if (op == "attn_k") {
            mapped_op = "wk";
        } else if (op == "attn_v") {
            mapped_op = "wv";
        } else if (op == "attn_output") {
            mapped_op = "wo";
        } else if (op == "ffn_norm") {
            mapped_op = "ffn_norm";
        } else if (op == "ffn_gate") {
            mapped_op = "w_gate";
        } else if (op == "ffn_up") {
            mapped_op = "w_up";
        } else if (op == "ffn_down") {
            mapped_op = "w_down";
        } else {
            return "";
        }
        return "layers." + layer_str + "." + mapped_op;
    }

    return "";
}

std::vector<int> MapShape(const std::vector<uint64_t>& gguf_shape) {
    std::vector<int> shape;
    shape.reserve(gguf_shape.size());
    if (gguf_shape.size() == 2) {
        shape.push_back(static_cast<int>(gguf_shape[1]));
        shape.push_back(static_cast<int>(gguf_shape[0]));
        return shape;
    }

    for (auto it = gguf_shape.rbegin(); it != gguf_shape.rend(); it++) {
        shape.push_back(static_cast<int>(*it));
    }

    return shape;
}

QuantizedTensor GetQuantTensor(const std::vector<uint8_t>& tensor, const std::vector<int>& shape, int quant_type) {
    size_t element_size = 1;
    for (auto& dim : shape) {
        element_size *= static_cast<size_t>(dim);
    }

    QuantizedTensor quant_tensor;
    quant_tensor.shape = shape;

    switch (quant_type) {
        case kQuantTypeF32: {
            quant_tensor.type = QuantType::kF32;
            size_t use_bytes = element_size * sizeof(float);
            if (tensor.size() != use_bytes) {
                throw std::runtime_error("f32 tensor size is not match");
            }
            quant_tensor.f32_data.resize(element_size);
            std::memcpy(quant_tensor.f32_data.data(), tensor.data(), use_bytes);
            break;
        }
        case kQuantTypeQ80: {
            quant_tensor.type = QuantType::kQ80;
            size_t num_blocks = (element_size + kQ80BlockSize - 1) / kQ80BlockSize;
            size_t use_bytes = num_blocks * sizeof(BlockQ80);
            if (tensor.size() != use_bytes) {
                throw std::runtime_error("Q80 tensor is not match");
            }
            quant_tensor.q8_0_data.resize(num_blocks);
            std::memcpy(quant_tensor.q8_0_data.data(), tensor.data(), use_bytes);
            break;
        }
        case kQuantTypeQ40: {
            quant_tensor.type = QuantType::kQ40;
            size_t num_blocks = (element_size + kQ40BlockSize - 1) / kQ40BlockSize;
            size_t use_bytes = num_blocks * sizeof(BlockQ40);
            if (tensor.size() != use_bytes) {
                throw std::runtime_error("q40 tensor size is not match");
            }
            quant_tensor.q4_0_data.resize(num_blocks);
            std::memcpy(quant_tensor.q4_0_data.data(), tensor.data(), use_bytes);
            break;
        }
        case kQuantTypeQ41: {
            quant_tensor.type = QuantType::kQ41;
            size_t num_blocks = (element_size + kQ41BlockSize - 1) / kQ41BlockSize;
            size_t use_bytes = num_blocks * sizeof(BlockQ41);
            if (tensor.size() != use_bytes) {
                throw std::runtime_error("q41 tensor size is not match");
            }
            quant_tensor.q4_1_data.resize(num_blocks);
            std::memcpy(quant_tensor.q4_1_data.data(), tensor.data(), use_bytes);
            break;          
        }
        default: {
            throw std::runtime_error("not find this type : " + std::to_string(quant_type));
        }
    }

    return quant_tensor;
}

void CheckTensorShape(const Tensor& tensor, const std::vector<int>& expect_shape, const std::string& caller) {
    tensor.CheckShape(expect_shape, caller.c_str());
}

void CheckTensorShape(const QuantizedTensor& q_tensor, const std::vector<int>& expect_shape, const std::string& caller) {
    q_tensor.AssertShape(expect_shape, caller.c_str());
}

void CheckOptionalShape(const Tensor& tensor, const std::vector<int>& expect_shape, const std::string& caller) {
    if (tensor.data.empty()) {
        return ;
    }
    tensor.CheckShape(expect_shape, caller.c_str());
}

void ValidateModel(const MiniLlamaModel& model) {
    const ModelConfig& config = model.config;
    const int q_dim = config.n_heads * config.head_dim;
    const int kv_dim = config.n_kv_heads * config.head_dim;

    CheckTensorShape(model.token_embedding, {config.vocab_size, config.dim}, "GGuf token_embedding");
    CheckTensorShape(model.final_norm, {config.dim}, "GGuf final_norm");
    CheckTensorShape(model.lm_head, {config.vocab_size, config.dim}, "GGuf lm_head");
    if (model.layers.size() != config.n_layers) {
        throw std::runtime_error("GGuf layers is not match from model to config");
    }

    for (int layer = 0; layer < config.n_layers; layer++) {
        const std::string prefix = "GGuf layer." + std::to_string(layer) + '.';
        const LayerWeights& lw = model.layers[layer];
        CheckTensorShape(lw.attention_norm, {config.dim}, prefix + "attention_norm");
        CheckTensorShape(lw.wq, {q_dim, config.dim}, prefix + "wq");
        CheckTensorShape(lw.wk, {kv_dim, config.dim}, prefix + "wk");
        CheckTensorShape(lw.wv, {kv_dim, config.dim}, prefix + "wv");
        CheckOptionalShape(lw.bq, {q_dim}, prefix + "bq");
        CheckOptionalShape(lw.bk, {kv_dim}, prefix + "bk");
        CheckOptionalShape(lw.bv, {kv_dim}, prefix + "bv");
        CheckTensorShape(lw.wo, {config.dim, q_dim}, prefix + "wo");
        CheckTensorShape(lw.ffn_norm, {config.dim}, prefix + "ffn_norm");
        CheckTensorShape(lw.w_gate, {config.hidden_dim, config.dim}, prefix + "w_gate");
        CheckTensorShape(lw.w_up, {config.hidden_dim, config.dim}, prefix + "w_up");
        CheckTensorShape(lw.w_down, {config.dim, config.hidden_dim}, prefix + "w_down");
    }
}

void ValidateConfig(ModelConfig& config) {
    if (config.n_layers <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
        config.n_heads <= 0 || config.n_kv_heads <= 0 ||
        config.max_seq_len <= 0 || config.vocab_size <= 0) {
        throw std::runtime_error(
            "Invalid model config from GGUF: dim=" + std::to_string(config.dim) +
            " layers=" + std::to_string(config.n_layers) +
            " heads=" + std::to_string(config.n_heads) +
            " vocab=" + std::to_string(config.vocab_size));
    }
    if (config.dim % config.n_heads != 0) {
        throw std::runtime_error("dim not divisible by n_heads");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::runtime_error("n_heads not divisible by n_kv_heads");
    }

}

void PrintConfig(const ModelConfig& config) {
    std::cout << "GGUF config: vocab=" << config.vocab_size
                << " dim=" << config.dim << " layers=" << config.n_layers
                << " heads=" << config.n_heads << " kv_heads=" << config.n_kv_heads
                << " head_dim=" << config.head_dim
                << " hidden_dim=" << config.hidden_dim
                << " max_seq_len=" << config.max_seq_len
                << " rope_theta=" << config.rope_theta
                << " rms_norm_eps=" << config.rms_norm_eps << std::endl;
}


ModelConfig ParseConfig(const std::string& path) {
    auto get_u32 = [&](const std::string& key, int default_data) {
        int64_t v = 0;
        if (GetGgufMetaDataInt(path, key, v)) {
            return static_cast<int>(v);
        }
        return default_data;
    };
    auto get_f32 = [&](const std::string& key, float default_data) {
        double v = 0.0;
        if (GetGgufMetaDataFloat(path, key, v)) {
            return static_cast<float>(v);
        }
        return default_data;
    };

    ModelConfig config;
    std::string model_name;
    if (GetGgufMetaDataString(path, "general.architecture", model_name) && model_name == "qwen2") {
        config.rope_type = RopeType::kNeoX;
    }

    config.n_layers = get_u32("qwen2.block_count", get_u32("llama.block_count", 0));
    config.dim = get_u32("qwen2.embedding_length", get_u32("llama.embedding_length", 0));
    config.hidden_dim = get_u32("qwen2.feed_forward_length", get_u32("llama.feed_forward_length", 0));
    config.n_heads = get_u32("qwen2.attention.head_count", get_u32("llama.attention.head_count", 0));
    config.n_kv_heads = get_u32("qwen2.attention.head_count_kv", get_u32("llama.attention.head_count_kv", 0));
    config.max_seq_len = get_u32("qwen2.context_length", get_u32("llama.context_length", 0));
    config.rope_theta = get_f32("qwen2.rope.freq_base", get_f32("llama.rope.freq_base", 10000.0f));
    config.rms_norm_eps = get_f32("qwen2.attention.layer_norm_rms_epsilon", get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f));

    std::vector<std::string> tokens;
    if (GetGgufMetaDataStringArray(path, "tokenizer.ggml.tokens", tokens)) {
        config.vocab_size = static_cast<int>(tokens.size());
    }

    ValidateConfig(config);
    config.head_dim = config.dim / config.n_heads;
//    PrintConfig(config);

    return config;  
}

QuantizedTensor LoadQuantWeight(const std::string& gguf_path, GgufReader& reader, std::map<std::string, GgufTensorInfo*>& tensorinfo_map, const std::string& name) {
    auto it = tensorinfo_map.find(name);
    if (it == tensorinfo_map.end()) {
        throw std::runtime_error("not find tensro :" + name);
    }

    GgufTensorInfo info = *it->second;
    std::vector<int> shape = MapShape(info.shape);
    std::vector<uint8_t> tensor = GetGgufSingleTensor(gguf_path, info, reader.data_offset);
    if (tensor.empty() && info.quant_type != kQuantTypeF32) {
        throw std::runtime_error("fail to read tensor:" + info.name);
    }
    return GetQuantTensor(tensor, shape, info.quant_type);
}

Tensor LoadF32Weight(const std::string& gguf_path, GgufReader& reader, std::map<std::string, GgufTensorInfo*>& tensorinfo_map, const std::string& name) {
    QuantizedTensor q_tensor = LoadQuantWeight(gguf_path, reader, tensorinfo_map, name);
    switch (q_tensor.type) {
        case QuantType::kF32: {
            return ToTensor(q_tensor);
        }
        case QuantType::kQ80: {
            return DeQuantizeFromQ80(q_tensor.q8_0_data, q_tensor.shape);
        }
        case QuantType::kQ40: {
            return DeQuantizeFromQ40(q_tensor.q4_0_data, q_tensor.shape);
        }
        case QuantType::kQ41: {
            return DequantizeFromQ41(q_tensor.q4_1_data, q_tensor.shape);
        }
    }
    throw std::runtime_error("quant type is not match: " + name);
}

Tensor LoadOptionalF32Weight(const std::string& gguf_path, GgufReader& reader, std::map<std::string, GgufTensorInfo*>& tensorinfo_map, const std::string& name) {
    if (tensorinfo_map.find(name) == tensorinfo_map.end()) {
        return Tensor();
    }
    return LoadF32Weight(gguf_path, reader, tensorinfo_map, name);
}




} //namespace 

MiniLlamaModel LoadGgufModel(const std::string& gguf_path) {
    MiniLlamaModel model;
    GgufReader reader;
    if (!reader.Load(gguf_path)) {
        model.load_error = "GGuf parse error: " + reader.load_error;
        return model;
    }

    try {
        model.config = ParseConfig(gguf_path);
    } catch (const std::exception& e) {
        model.load_error = std::string("Fail to build config: ") + e.what();
        return model;
    }

    PrintConfig(model.config);

    std::map<std::string, GgufTensorInfo*> tensorinfo_map;
    for (auto& info : reader.tensor_infos) {
        std::string use_name = MapTensorName(info.name);
        if (use_name.empty()) {
            continue;
        }
        tensorinfo_map[use_name] = &info;
    }

    try {
        model.token_embedding = LoadF32Weight(gguf_path, reader, tensorinfo_map, "token_embedding");
        //layers
        model.layers.resize(model.config.n_layers);
        for (int layer = 0; layer < model.config.n_layers; layer++) {
            LayerWeights& lw = model.layers[layer];
            const std::string prefix = "layers." + std::to_string(layer) + '.';
            lw.attention_norm = LoadF32Weight(gguf_path, reader, tensorinfo_map, prefix + "attention_norm");
            lw.wq = LoadQuantWeight(gguf_path, reader, tensorinfo_map, prefix + "wq");
            lw.wk = LoadQuantWeight(gguf_path, reader, tensorinfo_map, prefix + "wk");
            lw.wv = LoadQuantWeight(gguf_path, reader, tensorinfo_map, prefix + "wv");
            lw.bq = LoadOptionalF32Weight(gguf_path, reader, tensorinfo_map, prefix + "bq");
            lw.bk = LoadOptionalF32Weight(gguf_path, reader, tensorinfo_map, prefix + "bk");
            lw.bv = LoadOptionalF32Weight(gguf_path, reader, tensorinfo_map, prefix + "bv");
            lw.wo = LoadQuantWeight(gguf_path, reader, tensorinfo_map, prefix + "wo");
            lw.ffn_norm = LoadF32Weight(gguf_path, reader, tensorinfo_map, prefix + "ffn_norm");
            lw.w_gate = LoadQuantWeight(gguf_path, reader, tensorinfo_map, prefix + "w_gate");
            lw.w_up = LoadQuantWeight(gguf_path, reader, tensorinfo_map, prefix + "w_up");
            lw.w_down = LoadQuantWeight(gguf_path, reader, tensorinfo_map, prefix + "w_down");            
        }
        model.final_norm = LoadF32Weight(gguf_path, reader, tensorinfo_map, "final_norm");
        
        auto it = tensorinfo_map.find("lm_head");
        if (it != tensorinfo_map.end()) {
            model.lm_head = LoadQuantWeight(gguf_path, reader, tensorinfo_map, "lm_head");
        } else {
            model.lm_head = ToQuantizedTensor(model.token_embedding);
        }

        ValidateModel(model);
    } catch (const std::exception& e) {
        model.load_error = std::string("fail to load gguf weights") + e.what();
        return model;
    }

    model.is_loaded = true;
    std::cout << "GGuf model load successfully from " << gguf_path << std::endl;
    return model;
}



}