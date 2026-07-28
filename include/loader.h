#pragma once 


#include "include/model.h"

namespace mini_llama {

struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string data_type;
    size_t offset;
    size_t byte_size;
};

struct TokenizerInfo {
    std::string type = "ascii";
    std::string path;
    int bos_id = 1;
    int eos_id = 2;
    int unk_id = 0;
};

struct ModelManifest {
    ModelConfig config;
    TokenizerInfo tokenizer;
    std::vector<TensorInfo> tensor_infos;
};

ModelManifest ParseManifest(const std::string& config_path);

MiniLlamaModel LoadModel(const std::string& config_path, const std::string& weight_path);


}