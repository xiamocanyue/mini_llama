#include "include/ops.h"

#include <exception>
#include <stdexcept>
#include <math.h>

#include "include/matmul_dispatch.h"
#include "include/quant.h"

namespace mini_llama {

Tensor Matmul(const Tensor& a, const Tensor& b) {
    return MatulDispatch(a, b, DefalutMatulMode());
}

//逐个元素相乘
Tensor ElementMutmul(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("ElementMutmul: shape is not match");
    }
    Tensor y(a.shape, 0.0f);
    for (int i = 0; i < a.Size(); i++) {
        y.data[i] = a.data[i] * b.data[i];
    }
    return y;
}

//根据x的行数来返回相同行数的结果
Tensor Linear(const Tensor& x, const Tensor& weight) {
    return LinearDispatch(x, weight, DefalutMatulMode());
}

//根据x的shape来返回相同shape的结果
Tensor Linear(const Tensor& x, const QuantizedTensor& weight) {
    switch (weight.type) {
    case QuantType::kF32:
        return Linear(x, ToTensor(weight));
    case QuantType::kQ80:
        return LinearQ80(x, weight.q8_0_data, weight.shape);
    case QuantType::kQ40:
        return LinearQ40(x, weight.q4_0_data, weight.shape);
    case QuantType::kQ41:
        return LinearQ41(x, weight.q4_1_data, weight.shape);
    }
    throw std::runtime_error("Linear : unknow quant type");
}

// y = x / sqrt(mean(x^2) + eps) * weight
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.DimNum() != 1 || weight.DimNum() != 1) {
        throw std::runtime_error("RmsNorm: tensor is not 1D");
    }
    int dim = x.shape[0];
    if (weight.shape[0] != dim) {
        throw std::runtime_error("RmsNorm: dimension not match");
    }
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) {
        ss += x.data[i] * x.data[i];
    }
    ss = ss / static_cast<float>(dim);
    float scale = 1.0f / (std::sqrt(ss + eps));

    Tensor y({dim}, 0.0f);
    for (int i = 0; i < dim; i++) {
        y.data[i] = x.data[i] * scale * weight.data[i];
    }
    return y;
}

//e^(x - max) / sum(e^(x - max))
Tensor Softmax(const Tensor& x) {
    if (x.DimNum() != 1) {
        throw std::runtime_error("Softmax: Tensor is not 1D");
    }
    if (x.Size() == 0) {
        throw std::runtime_error("Softmax: tensor is empty");
    }
    int n = x.shape[0];
    float max_num = x.data[0];
    for (int i = 1; i < n; i++) {
        if (x.data[i] > max_num) {
            max_num = x.data[i];
        }
    }
    float sum = 0.0f;
    Tensor y({n}, 0.0f);
    for (int i = 0; i < n; i++) {
        y.data[i] = std::exp(x.data[i] - max_num);
        sum += y.data[i];
    }
    for (int i = 0; i < n; i++) {
        y.data[i] /= sum;
    }
    return y;
}

// x * sigmoid(x)
Tensor Silu(const Tensor& x) {
    Tensor y(x.shape, 0.0f);
    for (int i = 0; i < x.Size(); i++) {
        float val = x.data[i];
        y.data[i] = val / (1.0f + std::exp(-val));
    }
    return y;
}

//Swiglu = Silu(gate) · up = gate * sigmoid(gate) * up
Tensor Swiglu(const Tensor& gate, const Tensor& up) {
    if (gate.shape != up.shape) {
        throw std::runtime_error("Swiglu: shape is not match");
    }
    Tensor y(gate.shape, 0.0f);
    for (int i = 0; i < gate.Size(); i++) {
        float val = gate.data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-val));
        y.data[i] = val * sigmoid * up.data[i];
    }
    return y;
}


static void ApplyNormalRoPE(Tensor& x, int pos, const std::vector<float>& cos_table,
                           const std::vector<float>& sin_table) {
    int rows = x.shape[0];
    int row_length = x.shape[1];
    int half_length = row_length / 2;
    size_t table_offset = static_cast<size_t>(pos * half_length);
    for (int r = 0; r < rows; r++) {
        int row_begin = r * row_length;
        for (int i = 0; i < half_length; i++) {
            int index = 2 * i;

            float cos_val = cos_table[table_offset + i];
            float sin_val = sin_table[table_offset + i];

            float x0 = x.data[row_begin + index];
            float x1 = x.data[row_begin + index + 1];

            x.data[row_begin + index] = x0 * cos_val - x1 * sin_val;
            x.data[row_begin + index + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

static void ApplyNeoxRoPE(Tensor& x, int pos, const std::vector<float>& cos_table,
                           const std::vector<float>& sin_table) {
    int rows = x.shape[0];
    int row_length = x.shape[1];
    int half_length = row_length / 2;
    size_t table_offset = static_cast<size_t>(pos * half_length);
    for (int r = 0; r < rows; ++r) {
        int row_begin = r * row_length;
        for (int i = 0; i < half_length; i++) {
            int first = i;
            int second = half_length + i;

            float cos_val = cos_table[table_offset + i];
            float sin_val = sin_table[table_offset + i];

            float x0 = x.data[row_begin + first];
            float x1 = x.data[row_begin + second];

            x.data[row_begin + first] = x0 * cos_val - x1 * sin_val;
            x.data[row_begin + second] = x0 * sin_val + x1 * cos_val;
        }
    }
}

//freq = 1 / theta^(2 * i / head_dim)
//x1` = x1 * cos(m * freq) - x2 * sin(m * freq)
//x2` = x1 * sin(m * freq) + x2 * cos(m * freq)
void RoPE(Tensor& q, Tensor& k, int pos, const std::vector<float>& cos_table, const std::vector<float>& sin_table, RopeType rope_type) {
    if (q.DimNum() != 2 || k.DimNum() != 2) {
        throw std::runtime_error("RoPE : Tensor is not 2D");
    }
    if (pos < 0 || q.shape[1] <= 0 || k.shape[1] <= 0 ||
        q.shape[1] % 2 != 0 || k.shape[1] % 2 != 0 ||
        q.shape[1] != k.shape[1]) {
        throw std::runtime_error("RoPE: invalid tensor shape or position");
    }
    int half_dim = q.shape[1] / 2;
    size_t offset = static_cast<size_t>(pos * half_dim);
    if (offset + half_dim > cos_table.size() || offset + half_dim > sin_table.size()) {
        throw std::runtime_error("RoPE: position exceeds precomputed table");
    }

    if (rope_type == RopeType::kNormal) {
        ApplyNormalRoPE(q, pos, cos_table, sin_table);
        ApplyNormalRoPE(k, pos, cos_table, sin_table);
    } else {
        ApplyNeoxRoPE(q, pos, cos_table, sin_table);
        ApplyNeoxRoPE(k, pos, cos_table, sin_table);
    }
}



int GetArgMax(const Tensor& x) {
    if (x.Size() == 0) {
        throw std::runtime_error("GetArgMax : tensor is empty");
    }
    int index = 0;
    float max_num = x.data[0];
    for (size_t i = 0; i < x.data.size(); i++) {
        if (x.data[i] > max_num) {
            index = static_cast<int>(i);
            max_num = x.data[i];
        }
    }
    return index;
}

}
