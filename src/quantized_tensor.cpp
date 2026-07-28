#include "include/quantized_tensor.h"

namespace mini_llama {

Tensor ToTensor(const QuantizedTensor& qtensor) {
    if (qtensor.type != QuantType::kF32) {
        throw std::runtime_error("ToSensor: QuantizedTensor is not kF32");
    }
    Tensor t(qtensor.shape, 0.0f);
    if (qtensor.f32_data.size() == t.data.size()) {
        t.data = qtensor.f32_data;
    }
    return t;
}

QuantizedTensor ToQuantizedTensor(const Tensor& tensor) {
    QuantizedTensor q;
    q.type = QuantType::kF32;
    q.shape = tensor.shape;
    q.f32_data = tensor.data;
    return q;
}
}