#include "include/model.h"

#include <quant.h>
namespace mini_llama {

static size_t TensorBytes(const Tensor& t) {
    return t.data.size() * sizeof(float);
}

static size_t QuantizedTensorBytes(const QuantizedTensor& qt) {
    switch (qt.type) {
        case QuantType::kF32:
            return qt.f32_data.size() * sizeof(float);
        case QuantType::kQ80:
            return qt.q8_0_data.size() * sizeof(BlockQ80);
        case QuantType::kQ40:
            return qt.q4_0_data.size() * sizeof(BlockQ40);
        case QuantType::kQ41:
            return qt.q4_1_data.size() * sizeof(BlockQ41);
    }
    return 0;
}

static size_t QuantizedTensorBytesF32(const QuantizedTensor& qt) {
    return qt.ElementsNum() * sizeof(float);
}

size_t ModelWeightBytes(const MiniLlamaModel& model) {
    size_t bytes = 0;
    bytes += TensorBytes(model.token_embedding);
    bytes += TensorBytes(model.final_norm);
    bytes += QuantizedTensorBytes(model.lm_head);
    for (const auto& lw : model.layers) {
        bytes += TensorBytes(lw.attention_norm);
        bytes += QuantizedTensorBytes(lw.wq);
        bytes += QuantizedTensorBytes(lw.wk);
        bytes += QuantizedTensorBytes(lw.wv);
        bytes += TensorBytes(lw.bq);
        bytes += TensorBytes(lw.bk);
        bytes += TensorBytes(lw.bv);
        bytes += QuantizedTensorBytes(lw.wo);
        bytes += TensorBytes(lw.ffn_norm);
        bytes += QuantizedTensorBytes(lw.w_gate);
        bytes += QuantizedTensorBytes(lw.w_up);
        bytes += QuantizedTensorBytes(lw.w_down);
    }
    return bytes;
}

size_t ModelWeightBytesF32(const MiniLlamaModel& model) {
    size_t bytes = 0;
    bytes += TensorBytes(model.token_embedding);
    bytes += TensorBytes(model.final_norm);
    bytes += QuantizedTensorBytesF32(model.lm_head);
    for (const auto& lw : model.layers) {
        bytes += TensorBytes(lw.attention_norm);
        bytes += QuantizedTensorBytesF32(lw.wq);
        bytes += QuantizedTensorBytesF32(lw.wk);
        bytes += QuantizedTensorBytesF32(lw.wv);
        bytes += TensorBytes(lw.bq);
        bytes += TensorBytes(lw.bk);
        bytes += TensorBytes(lw.bv);
        bytes += QuantizedTensorBytesF32(lw.wo);
        bytes += TensorBytes(lw.ffn_norm);
        bytes += QuantizedTensorBytesF32(lw.w_gate);
        bytes += QuantizedTensorBytesF32(lw.w_up);
        bytes += QuantizedTensorBytesF32(lw.w_down);
    }
    return bytes;
}

static void UploadF32LinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight,
                                  const std::string& name, int device_id) {
    if (weight.type != QuantType::kF32) {
        return;
    }
    if (weight.f32_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty F32 data for " + name);
    }

    const size_t expected_numel = weight.ElementsNum();
    if (weight.f32_data.size() != expected_numel) {
        throw std::runtime_error("CUDA weight upload: data size mismatch for " +
                                name + ", expected " +
                                std::to_string(expected_numel) + ", got " +
                                std::to_string(weight.f32_data.size()));
    }

    const size_t bytes = weight.f32_data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kF32;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.f32_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadF32TensorWeight(CudaModelWeights& storage, const Tensor& weight, 
                                const std::string& name, int device_id) {
    if (weight.data.empty()) {
        return;
    }
    const size_t expected_numel = weight.Size();
    if (weight.data.size() != expected_numel) {
        throw std::runtime_error(
            "CUDA weight upload: tensor data size mismatch for " + name +
            ", expected " + std::to_string(expected_numel) + ", got " +
            std::to_string(weight.data.size()));
    }

    const size_t bytes = weight.data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kF32;
    uploaded.linear = false;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadQ80LinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight,
                                  const std::string& name, int device_id) {
    if (weight.type != QuantType::kQ80) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error(
            "CUDA weight upload: expected 2D Q8_0 weight for " + name);
    }
    if (weight.q8_0_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q8_0 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q8_0_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q8_0 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) + ", got " +
            std::to_string(weight.q8_0_data.size()));
    }

    const size_t bytes = weight.q8_0_data.size() * sizeof(BlockQ80);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ80;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q8_0_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q8_0_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadQ40LinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight,
                                  const std::string& name, int device_id) {
    if (weight.type != QuantType::kQ40) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error("CUDA weight upload: expected 2D Q4_0 weight for " + name);
    }
    if (weight.q4_0_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q4_0 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q4_0_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q4_0 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) + ", got " +
            std::to_string(weight.q4_0_data.size()));
    }

    const size_t bytes = weight.q4_0_data.size() * sizeof(BlockQ40);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ40;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q4_0_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q4_0_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadQ41LinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight,
                                  const std::string& name, int device_id) {
    if (weight.type != QuantType::kQ41) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error("CUDA weight upload: expected 2D Q4_1 weight for " + name);
    }
    if (weight.q4_1_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q4_1 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ41BlockSize - 1) / kQ41BlockSize;
    const size_t expected_blocks = static_cast<size_t>(out_features) * blocks_per_row;
    if (weight.q4_1_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q4_1 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) + ", got " +
            std::to_string(weight.q4_1_data.size()));
    }

    const size_t bytes = weight.q4_1_data.size() * sizeof(BlockQ41);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ41;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q4_1_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q4_1_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadLinearWeight(CudaModelWeights& storage, const QuantizedTensor& weight,
                               const std::string& name, int device_id) {
    UploadF32LinearWeight(storage, weight, name, device_id);
    UploadQ80LinearWeight(storage, weight, name, device_id);
    UploadQ40LinearWeight(storage, weight, name, device_id);
    UploadQ41LinearWeight(storage, weight, name, device_id);
}

void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id) {
    if (!model.is_loaded) {
        throw std::runtime_error("CUDA weight upload requires a loaded model");
    }

    CudaSetDevice(device_id);
    auto storage = std::make_shared<CudaModelWeights>();
    storage->device_id = device_id;

    UploadF32TensorWeight(*storage, model.token_embedding, "token_embedding", device_id);
    UploadF32TensorWeight(*storage, model.final_norm, "final_norm", device_id);
    UploadLinearWeight(*storage, model.lm_head, "lm_head", device_id);

    for (size_t layer = 0; layer < model.layers.size(); ++layer) {
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        UploadF32TensorWeight(*storage, lw.attention_norm, prefix + "attention_norm", device_id);
        UploadLinearWeight(*storage, lw.wq, prefix + "wq", device_id);
        UploadLinearWeight(*storage, lw.wk, prefix + "wk", device_id);
        UploadLinearWeight(*storage, lw.wv, prefix + "wv", device_id);
        UploadF32TensorWeight(*storage, lw.bq, prefix + "bq", device_id);
        UploadF32TensorWeight(*storage, lw.bk, prefix + "bk", device_id);
        UploadF32TensorWeight(*storage, lw.bv, prefix + "bv", device_id);
        UploadLinearWeight(*storage, lw.wo, prefix + "wo", device_id);
        UploadF32TensorWeight(*storage, lw.ffn_norm, prefix + "ffn_norm", device_id);
        UploadLinearWeight(*storage, lw.w_gate, prefix + "w_gate", device_id);
        UploadLinearWeight(*storage, lw.w_up, prefix + "w_up", device_id);
        UploadLinearWeight(*storage, lw.w_down, prefix + "w_down", device_id);
    }

    model.cuda_weights = std::move(storage);
}

void ClearModelCudaWeights(MiniLlamaModel& model) {
    model.cuda_weights.reset();
}

bool ModelHasCudaWeights(const MiniLlamaModel& model) {
    return model.cuda_weights != nullptr;
}

size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->uploaded_weight_count;
}

static size_t ModelCudaUploadedWeightCountByType(const MiniLlamaModel& model, QuantType type) {
    if (!model.cuda_weights) {
        return 0;
    }
    size_t count = 0;
    for (const auto& weight : model.cuda_weights->weights) {
        if (weight.linear && weight.type == type) {
            count += 1;
        }
    }
    return count;
}

size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model) {
    return ModelCudaUploadedWeightCountByType(model, QuantType::kF32);
}

size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model) {
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ80);
}

size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model) {
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ40);
}

size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model) {
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ41);
}

size_t ModelCudaMemoryBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->uploaded_bytes;
}

void ResetModelCudaRuntimeStats(MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->linear_calls = 0;
    model.cuda_weights->activation_calls = 0;
    model.cuda_weights->attention_calls = 0;
    model.cuda_weights->attention_cpu_fallbacks = 0;
    model.cuda_weights->kv_cache_write_bytes = 0;
    model.cuda_weights->kv_cache_read_bytes = 0;
    model.cuda_weights->host_to_device_copies = 0;
    model.cuda_weights->device_to_host_copies = 0;
    model.cuda_weights->host_to_device_bytes = 0;
    model.cuda_weights->device_to_host_bytes = 0;
}

size_t ModelCudaLinearCalls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->linear_calls;
}

size_t ModelCudaActivationCalls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->activation_calls;
}

size_t ModelCudaAttentionCalls(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->attention_calls;
}

size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->attention_cpu_fallbacks;
}

size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->kv_cache_write_bytes;
}

size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->kv_cache_read_bytes;
}

size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->host_to_device_copies;
}

size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->device_to_host_copies;
}

size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->host_to_device_bytes;
}

size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->device_to_host_bytes;
}

static Tensor QuantizedTensorToF32(const QuantizedTensor& qt) {
    switch (qt.type) {
        case QuantType::kF32:
            return ToTensor(qt);
        case QuantType::kQ80:
            return DeQuantizeFromQ80(qt.q8_0_data, qt.shape);
        case QuantType::kQ40:
            return DeQuantizeFromQ40(qt.q4_0_data, qt.shape);
        case QuantType::kQ41:
            return DequantizeFromQ41(qt.q4_1_data, qt.shape);
    }
    throw std::runtime_error("quantized_tensor_to_f32: unknown quant type");
}

static void QuantizeQtToQ80(QuantizedTensor& qt) {
    if (qt.type == QuantType::kQ80) {
        return;
    }
    Tensor t = QuantizedTensorToF32(qt);
    qt.q8_0_data = QuantizeToQ80(t);
    qt.type = QuantType::kQ80;
    qt.f32_data.clear();
    qt.f32_data.shrink_to_fit();
    qt.q4_0_data.clear();
    qt.q4_0_data.shrink_to_fit();
    qt.q4_1_data.clear();
    qt.q4_1_data.shrink_to_fit();
}

void QuantizeModelToQ80(MiniLlamaModel& model) {
    QuantizeQtToQ80(model.lm_head);
    for (auto& lw : model.layers) {
        QuantizeQtToQ80(lw.wq);
        QuantizeQtToQ80(lw.wk);
        QuantizeQtToQ80(lw.wv);
        QuantizeQtToQ80(lw.wo);
        QuantizeQtToQ80(lw.w_gate);
        QuantizeQtToQ80(lw.w_up);
        QuantizeQtToQ80(lw.w_down);
    }
}

static void QuantizeQtToQ40(QuantizedTensor& qt) {
    if (qt.type == QuantType::kQ40) {
        return;
    }
    Tensor t = QuantizedTensorToF32(qt);
    qt.q4_0_data = QuantizeToQ40(t);
    qt.type = QuantType::kQ40;
    qt.f32_data.clear();
    qt.f32_data.shrink_to_fit();
    qt.q8_0_data.clear();
    qt.q8_0_data.shrink_to_fit();
    qt.q4_1_data.clear();
    qt.q4_1_data.shrink_to_fit();
}

void QuantizeModelToQ40(MiniLlamaModel& model) {
    QuantizeQtToQ40(model.lm_head);
    for (auto& lw : model.layers) {
        QuantizeQtToQ40(lw.wq);
        QuantizeQtToQ40(lw.wk);
        QuantizeQtToQ40(lw.wv);
        QuantizeQtToQ40(lw.wo);
        QuantizeQtToQ40(lw.w_gate);
        QuantizeQtToQ40(lw.w_up);
        QuantizeQtToQ40(lw.w_down);
    }
}

}