#include "include/loader.h"

#include <cmath>
#include <sstream>
#include <fstream>
#include <map>
#include <iostream> 
#include <algorithm>

namespace mini_llama {
namespace {

constexpr auto kNotFindString = std::string::npos;

MiniLlamaModel LoadFailure(const std::string& message) {
    MiniLlamaModel model;
    model.is_loaded = false;
    model.load_error = message;
    std::cerr << "Failed to load model" << message << std::endl;
    return model;
}

std::string ShapeToString(const std::vector<int>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

void SkipSpace(const std::string& content, size_t& pos) {
    while (pos < content.size() && std::isspace(static_cast<char>(content[pos]))) {
        pos++;
    }
}


bool IsJsonValueTerminator(char c) {
    return std::isspace(c) || c == ',' || c == '}' || c == ']';
}

void ValidateNumberTerminator(const std::string& content, size_t value_pos, size_t use_size, const std::string& key) {
    size_t pos = value_pos + use_size;
    SkipSpace(content, pos);
    if (pos < content.size() && !IsJsonValueTerminator(content[pos])) {
        throw std::runtime_error("value is not valid from key : " + key);
    }
}

void ValidateConfig(const ModelConfig& config) {
    if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
        config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
        config.max_seq_len <= 0) {
        throw std::runtime_error("all model dimensions must be positive");
    }
    if (config.dim % config.n_heads != 0) {
        throw std::runtime_error("dim must be divisible by n_heads");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::runtime_error("n_heads must be divisible by n_kv_heads");
    }
    if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
        throw std::runtime_error("head_dim must be a positive even number");
    }
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
        throw std::runtime_error("rope_theta must be positive");
    }
    if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
        throw std::runtime_error("rms_norm_eps must be positive");
    }
}

size_t CheckAndGetTensorByteSize(const TensorInfo& info) {
    size_t num_elements = 1;
    for (int dim : info.shape) {
        if (dim <= 0) {
            throw std::runtime_error("tensor shape is not positive from : " + info.name);
        }
        if (num_elements > (std::numeric_limits<size_t>::max() / static_cast<size_t>(dim))) {
            throw std::runtime_error("tensor elements size is over size_t::max() from:" + info.name);
        }
        num_elements *= dim;
    }
    if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(float))) {
        throw std::runtime_error("tensor byte size is over size_t::max() from:" + info.name);
    }
    return num_elements * sizeof(float);
}

std::vector<std::pair<std::string, std::vector<int>>> GetExpectedModelTensorShapes(const ModelConfig& config) {
    std::vector<std::pair<std::string, std::vector<int>>> expected = {{"token_embedding", {config.vocab_size, config.dim}},
                                                                      {"final_norm", {config.dim}},
                                                                      {"lm_head", {config.vocab_size, config.dim}},};

    for (int layer = 0; layer < config.n_layers; layer++) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        expected.push_back({prefix + "attention_norm", {config.dim}});
        expected.push_back({prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
        expected.push_back({prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back({prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back({prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
        expected.push_back({prefix + "ffn_norm", {config.dim}});
        expected.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
        expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
        expected.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
    }
    return expected;
}


void ValidateManifest(const ModelManifest& manifest) {
    if (manifest.tensor_infos.empty()) {
        throw std::runtime_error("model.json contains no tensors");
    }

    std::map<std::string, TensorInfo> tensor_map;
    std::vector<TensorInfo> infos;
    infos.reserve(manifest.tensor_infos.size());

    for (auto& info : manifest.tensor_infos) {
        if (info.data_type != "float32") {
            throw std::runtime_error("unsupported data type" + info.data_type + "for tensor " + info.name);
        }
        if (info.shape.empty()) {
            throw std::runtime_error("tensor shape is empty" + info.name);
        }

        size_t expected_byte = CheckAndGetTensorByteSize(info);
        if (expected_byte != info.byte_size) {
            throw std::runtime_error("tensor byte_size is not match for " + info.name +
                                    "expected:" + std::to_string(expected_byte) +
                                    "manifest is read :" + std::to_string(info.byte_size));
        }

        if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
            throw std::runtime_error("tensor offset is overflow for " + info.name);
        }

        auto insert = tensor_map.emplace(info.name, info);
        if (!insert.second) {
            throw std::runtime_error("tensor is not allow same for " + info.name);
        }

        infos.push_back(info);
    }

    std::sort(infos.begin(), infos.end(), [](const TensorInfo& a, const TensorInfo& b) {
        return a.offset < b.offset;
    });

    for (size_t i = 1; i < infos.size(); i++) {
        size_t front_end = infos[i - 1].offset + infos[i - 1].byte_size;
        if (infos[i].offset < front_end) {
            throw std::runtime_error("tensor ranges is overlap " + infos[i - 1].name + "and" + infos[i].name);
        }
    }

    for (auto& [name, shape] : GetExpectedModelTensorShapes(manifest.config)) {
        auto it = tensor_map.find(name);
        if (it == tensor_map.end()) {
            throw std::runtime_error("not find required tensor" + name);
        }
        if (it->second.shape != shape) {
            throw std::runtime_error("tensor shape is not match for" + name + 
                                    "expected" + ShapeToString(shape) + 
                                    "manifest is have " + ShapeToString(it->second.shape));
        }
    }
}


size_t GetConfigValueIndexFormKey(const std::string& content, const std::string& key) {
    const std::string key_target = '\"' + key + '\"';
    size_t key_pos = content.find(key_target);
    if (key_pos == kNotFindString) {
        throw std::runtime_error("not find config key: " + key);
    }

    size_t colon_pos = content.find(":", key_target.size() + key_pos);
    if (colon_pos == kNotFindString) {
        throw std::runtime_error("not find key back colon :" + key);
    }

    size_t value_pos = colon_pos + 1;
    SkipSpace(content, value_pos);
    if (value_pos >= content.size()) {
        throw std::runtime_error("not find value from key:" + key);
    }
    return value_pos;
}    

size_t GetOpentialValueIndexFormKey(const std::string& content, const std::string& key) {
    const std::string key_target = '\"' + key + '\"';
    size_t key_pos = content.find(key_target);
    if (key_pos == kNotFindString) {
        return kNotFindString;
    }

    size_t colon_pos = content.find(":", key_target.size() + key_pos);
    if (colon_pos == kNotFindString) {
        throw std::runtime_error("not find key back colon :" + key);
    }

    size_t value_pos = colon_pos + 1;
    SkipSpace(content, value_pos);
    if (value_pos >= content.size()) {
        throw std::runtime_error("not find value from key:" + key);
    }
    return value_pos;
}

std::string GetObjectValueFromKey(const std::string& content, const std::string& key) {
    const std::string key_target = '\"' + key + '\"';
    size_t key_pos = content.find(key_target);
    if (key_pos == kNotFindString) {
        return "";
    }

    size_t obj_start = content.find('{', key_target.size() + key_pos);
    if (obj_start == kNotFindString) {
        throw std::runtime_error("not find value from key : " + key);
    }

    int count = 0;
    for (size_t p = obj_start; p < content.size(); p++) {
        if (content[p] == '{') {
            count++;
        } else if (content[p] == '}') {
            count--;
            if (count == 0) {
                return content.substr(obj_start, p - obj_start + 1);
            }
        }
    }

    throw std::runtime_error("not find obj end, not have } from key :" + key);
}

int ParseRequiredInt(const std::string& content, const std::string& key) {
    size_t value_pos = GetConfigValueIndexFormKey(content, key);
    size_t use_size = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &use_size);
    } catch (const std::exception&) {
        throw std::runtime_error("value is not interger from key : " + key);
    }

    if (use_size == 0) {
        throw std::runtime_error("value is not interger from key : " + key);
    }

    ValidateNumberTerminator(content, value_pos, use_size, key);
    return value;
}

float ParseRequiredFloat(const std::string& content, const std::string& key) {
    size_t value_pos = GetConfigValueIndexFormKey(content, key);
    size_t use_size = 0;
    float value = 0.0;
    try {
        value = std::stof(content.substr(value_pos), &use_size);
    } catch (const std::exception&) {
        throw std::runtime_error("value is not float from key : " + key);
    }

    if (use_size == 0) {
        throw std::runtime_error("value is not float from key : " + key);
    }

    ValidateNumberTerminator(content, value_pos, use_size, key);
    return value;
}

std::string ParseOptionalString(const std::string& content, const std::string& key, const std::string& default_data) {
    size_t value_pos = GetOpentialValueIndexFormKey(content, key);
    if (value_pos == kNotFindString) {
        return default_data;
    }

    if (content[value_pos] != '"') {
        throw std::runtime_error("value is not string from key : " + key);
    }
    size_t end = content.find('"', value_pos + 1);
    if (end == kNotFindString) {
        throw std::runtime_error("value is not have end from key: " + key);
    }
    return content.substr(value_pos + 1, end - value_pos - 1);
}

int ParseOptionalInt(const std::string& content, const std::string& key, const int default_data) {
    size_t value_pos = GetOpentialValueIndexFormKey(content, key);
    if (value_pos == kNotFindString) {
        return default_data;
    }
    size_t use_size = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &use_size);
    } catch (const std::exception&) {
        throw std::runtime_error("value is not interger from key : " + key);
    }

    if (use_size == 0) {
        throw std::runtime_error("value is not interger from key : " + key);
    }
    ValidateNumberTerminator(content, value_pos, use_size, key);
    return value;
}

TokenizerInfo ParseTokenizerJson(const std::string& content) {
    TokenizerInfo tokenizer;
    std::string obj = GetObjectValueFromKey(content, "tokenizer");
    if (obj.empty()) {
        return tokenizer;
    }

    tokenizer.type = ParseOptionalString(obj, "type", tokenizer.type);
    tokenizer.path = ParseOptionalString(obj, "path", tokenizer.path);
    tokenizer.bos_id = ParseOptionalInt(obj, "bos_id", tokenizer.bos_id);
    tokenizer.eos_id = ParseOptionalInt(obj, "eos_id", tokenizer.eos_id);
    tokenizer.unk_id = ParseOptionalInt(obj, "unk_id", tokenizer.unk_id);    

    if (tokenizer.type != "ascii" && tokenizer.type != "json_vocab") {
        throw std::runtime_error("unsupported tokenizer type: " + tokenizer.type);
    }
    if (tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0) {
        throw std::runtime_error("tokenizer special ids must be non-negative");
    }
    if (tokenizer.bos_id == tokenizer.eos_id ||
        tokenizer.bos_id == tokenizer.unk_id ||
        tokenizer.eos_id == tokenizer.unk_id) {
        throw std::runtime_error("tokenizer special ids must be distinct");
    }
    if (tokenizer.type == "json_vocab" && tokenizer.path.empty()) {
        throw std::runtime_error("json_vocab tokenizer requires a path");
    }

    return tokenizer;

}

ModelConfig ParseConfigJson(const std::string& content) {
    ModelConfig config;
    config.vocab_size = ParseRequiredInt(content, "vocab_size");
    config.dim = ParseRequiredInt(content, "dim");
    config.hidden_dim = ParseRequiredInt(content, "hidden_dim");
    config.n_layers = ParseRequiredInt(content, "n_layers");
    config.n_heads = ParseRequiredInt(content, "n_heads");
    config.n_kv_heads = ParseRequiredInt(content, "n_kv_heads");
    config.max_seq_len = ParseRequiredInt(content, "max_seq_len");
    config.rope_theta = ParseRequiredFloat(content, "rope_theta");
    config.rms_norm_eps = ParseRequiredFloat(content, "rms_norm_eps");
    config.head_dim = config.dim / config.n_heads;

    ValidateConfig(config);
    return config;
}

bool ParseTensorStringSingle(const std::string& obj, TensorInfo& info, const std::string& key) {
    const std::string& key_target = '\"' + key + '\"';
    size_t key_pos = obj.find(key_target);
    if (key_pos != kNotFindString) {
        size_t colon_pos = obj.find(':', key_pos);
        size_t value_start = obj.find('\"', colon_pos + 1);
        size_t value_end = obj.find('\"', value_start + 1);
        if (value_start != kNotFindString && value_end != kNotFindString) {
            if (key == "name") {
                info.name = obj.substr(value_start + 1, value_end - value_start - 1);
                return true;                
            } else if (key == "data_type" || key == "dtype") {
                info.data_type = obj.substr(value_start + 1, value_end - value_start - 1);
                return true;                
            }
        }
    }
    return false;
}

bool ParseTensorIntSingle(const std::string& obj, TensorInfo& info, const std::string& key) {
    const std::string& key_target = '\"' + key + '\"';
    size_t key_pos = obj.find(key_target);
    if (key_pos != kNotFindString) {
        size_t colon_pos = obj.find(':', key_pos);
        if (colon_pos == kNotFindString) {
            throw std::runtime_error("tensor is not have colon");
        }

        if (key == "offset") {
            info.offset = static_cast<size_t>(std::stoull(obj.substr(colon_pos + 1)));
            return true;
        } else if (key == "byte_size") {
            info.byte_size = static_cast<size_t>(std::stoull(obj.substr(colon_pos + 1)));
            return true;
        }
    }
    return false;
}

bool ParseTensorVectorSingle(const std::string& obj, TensorInfo& info, const std::string& key) {
    const std::string& key_target = '\"' + key + '\"';
    size_t key_pos = obj.find(key_target);
    if (key_pos != kNotFindString) {
        size_t vector_start = obj.find('[', key_pos);
        size_t vector_end = obj.find(']', vector_start);
        if (vector_start != kNotFindString && vector_end != kNotFindString) {
            std::string vector_string = obj.substr(vector_start + 1, vector_end - vector_start - 1);
            std::stringstream ss(vector_string);
            std::string num;
            while (std::getline(ss, num, ',')) {
                size_t start = num.find_first_not_of(" \t\n\r");
                size_t end = num.find_last_not_of(" \t\n\r");
                if (start != kNotFindString) {
                    int dim = std::stoi(num.substr(start, end - start + 1));
                    info.shape.push_back(dim);
                }
            }
            return true;
        }
    }
    return false;
}

/*   
"tensors": [
{
  "name": "layers.0.wq",
  "dtype": "float32",
  "shape": [8, 8],
  "offset": 1024,
  "byte_size": 256
},
{
  "name": "layers.0.wk",
  "dtype": "float32",
  "shape": [8, 8],
  "offset": 1024,
  "byte_size": 256
}]
  */ 
std::vector<TensorInfo> ParseTensorInfos(const std::string& content) {
    std::vector<TensorInfo> result;
    const std::string key = "\"tensors\"";
    size_t key_pos = content.find(key);
    if (key_pos == kNotFindString) {
        throw std::runtime_error("not find tensors int model.json");
    }

    size_t bracket_pos = content.find('[', key_pos);
    if (bracket_pos == kNotFindString) {
        throw std::runtime_error("not find '[' after tensors");
    }

    size_t i = bracket_pos + 1;
    while (i < content.size()) {
        while (i < content.size() && content[i] != '{') {
            if (content[i] == ']') {
                return result;
            }
            i++;
        }
        if (i >= content.size()) {
            break;
        }

        TensorInfo info;
        size_t obj_start = i;
        size_t obj_end = content.find('}', obj_start);
        if (obj_end == kNotFindString) {
            throw std::runtime_error("unclose tenor object");
        }
        std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

        bool has_name = ParseTensorStringSingle(obj, info, "name");
        bool has_dtype = ParseTensorStringSingle(obj, info, "data_type") ||
                 ParseTensorStringSingle(obj, info, "dtype");
        bool has_shape = ParseTensorVectorSingle(obj, info, "shape");
        bool has_offset = ParseTensorIntSingle(obj, info, "offset");
        bool has_byte_size = ParseTensorIntSingle(obj, info, "byte_size");   

        if (!has_name || info.name.empty()) {
            throw std::runtime_error("tensor missing name field");
        }
        if (!has_dtype) {
            throw std::runtime_error("tensor missing dtype field: " + info.name);
        }
        if (!has_shape) {
            throw std::runtime_error("tensor missing shape field: " + info.name);
        }
        if (!has_offset) {
            throw std::runtime_error("tensor missing offset field: " + info.name);
        }
        if (!has_byte_size) {
            throw std::runtime_error("tensor missing byte_size field: " + info.name);
        }
        result.push_back(info);
        i = obj_end + 1;
    }

    return result;
}

Tensor LoadTensorByName(std::ifstream& wf, const std::map<std::string, TensorInfo>& tensor_map, const std::string& name) {
    auto it = tensor_map.find(name);
    if (it == tensor_map.end()) {
        throw std::runtime_error("not find tensor name is" + name);
    }

    const TensorInfo& info = it->second;
    if (info.offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("tensor offset is over streampos limit for " + name);
    }
    if (info.byte_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("tensor byte size is over stream size limit for" + name);
    }

    wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
    if (!wf.good()) {
        throw std::runtime_error("fail to seek to tensor offset for " + name);
    }

    Tensor t(info.shape, 0.0f);
    wf.read(reinterpret_cast<char*>(t.data.data()), static_cast<std::streamsize>(info.byte_size));
    if (static_cast<size_t>(wf.gcount()) != info.byte_size) {
        throw std::runtime_error("read tensor is not finish for name: " + name);
    }

    return t;
}

MiniLlamaModel ParseModelTensors(MiniLlamaModel& model, const std::map<std::string, TensorInfo>& tensor_map, const ModelManifest& manifest, std::ifstream& wf) {
    model.config = manifest.config;
    try {
        model.token_embedding = LoadTensorByName(wf, tensor_map, "token_embedding");
        model.layers.resize(manifest.config.n_layers);
        for (int layer = 0; layer < manifest.config.n_layers; layer++) {
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            LayerWeights& lw = model.layers[layer];
            lw.attention_norm = LoadTensorByName(wf, tensor_map, prefix + "attention_norm");
            lw.wq = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "wq"));
            lw.wk = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "wk"));
            lw.wv = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "wv"));
            lw.wo = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "wo"));
            lw.w_gate = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "w_gate"));
            lw.w_up =ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "w_up"));
            lw.w_down = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, prefix + "w_down"));
            lw.ffn_norm = LoadTensorByName(wf, tensor_map, prefix + "ffn_norm");
        }
        model.final_norm = LoadTensorByName(wf, tensor_map, "final_norm");
        model.lm_head = ToQuantizedTensor(LoadTensorByName(wf, tensor_map, "lm_head"));
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }
    return model;
}

} // namespace


ModelManifest ParseManifest(const std::string& config_path) {
    ModelManifest manifest;
    std::ifstream cf(config_path);
    if (!cf.is_open()) {
        throw std::runtime_error("failed to open config: " + config_path);
    }
    std::stringstream buffer;
    buffer << cf.rdbuf();
    if (cf.bad()) {
        throw std::runtime_error("failed to read config: " + config_path);
    }

    std::string content = buffer.str();
    manifest.config = ParseConfigJson(content);
    manifest.tokenizer = ParseTokenizerJson(content);
    manifest.tensor_infos = ParseTensorInfos(content);
    ValidateManifest(manifest);

    return manifest;   
}

MiniLlamaModel LoadModel(const std::string& config_path, const std::string& weights_path) {
    ModelManifest manifest;
    try {
        manifest = ParseManifest(config_path);
    } catch(const std::exception& e) {
        return LoadFailure(e.what());
    }
    std::cout << "Loaded config: vocab=" << manifest.config.vocab_size
            << " dim=" << manifest.config.dim
            << " layers=" << manifest.config.n_layers
            << " heads=" << manifest.config.n_heads
            << " kv_heads=" << manifest.config.n_kv_heads
            << " head_dim=" << manifest.config.head_dim
            << " hidden_dim=" << manifest.config.hidden_dim << std::endl;
    
    std::map<std::string, TensorInfo> tensor_map;
    for (auto& info : manifest.tensor_infos) {
        tensor_map[info.name] = info;
    }

    std::ifstream wf(weights_path, std::ios::binary);
    if (!wf.is_open()) {
        return LoadFailure("failed to open weights: " + weights_path);
    }

    MiniLlamaModel model;
    ParseModelTensors(model, tensor_map, manifest, wf);

    wf.seekg(0, std::ios::end);
    std::streamoff file_size = wf.tellg();
    if (file_size < 0) {
        return LoadFailure("fail to determine weight file size");
    }

    size_t expected_end = 0;
    for (auto& info : manifest.tensor_infos) {
        size_t tensor_end = info.offset + info.byte_size;
        if (expected_end < tensor_end) {
            expected_end = tensor_end;
        }
    }

    if (static_cast<size_t>(file_size) != expected_end) {
        if (static_cast<size_t>(file_size) > expected_end) {
            return LoadFailure("weights file has trailing bytes: file size=" + std::to_string(file_size) +
                              ", expected=" + std::to_string(expected_end));
        } else {
            return LoadFailure("weights file is too short: file size=" + std::to_string(file_size) +
                              ", expected=" + std::to_string(expected_end));
        }
    }

    model.is_loaded = true;
    std::cout << "Model load successfully !!!!!!!!!!!" << std::endl;
    return model;
}

}