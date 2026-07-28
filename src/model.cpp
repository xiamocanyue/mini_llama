#include "include/model.h"


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


}