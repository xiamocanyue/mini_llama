#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "include/cuda_attention.h"
#include "include/cuda_data.h"
#include "include/cuda_kv_cache.h"
#include "include/cuda_matmul.h"
#include "include/cuda_ops.h"
#include "include/cuda_quant.h"
#include "include/cuda_tensor.h"
#include "include/forward.h"
#include "include/kv_cache.h"
#include "include/loader.h"
#include "include/ops.h"
#include "include/quant.h"

namespace mini_llama {
namespace {

constexpr float kFloatTolerance = 2e-4f;
constexpr float kQuantTolerance = 2e-3f;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireClose(const Tensor& actual, const Tensor& expected, float tolerance,
                  const std::string& label) {
    Require(actual.shape == expected.shape,
            label + ": shape mismatch " + actual.ShapeString() + " vs " +
                expected.ShapeString());
    for (size_t i = 0; i < actual.Size(); ++i) {
        Require(std::isfinite(actual.data[i]) && std::isfinite(expected.data[i]),
                label + ": non-finite value at " + std::to_string(i));
        float error = std::abs(actual.data[i] - expected.data[i]);
        float scale = std::max(1.0f, std::abs(expected.data[i]));
        if (error > tolerance && error / scale > tolerance) {
            throw std::runtime_error(
                label + ": mismatch at " + std::to_string(i) + ", actual=" +
                std::to_string(actual.data[i]) + ", expected=" +
                std::to_string(expected.data[i]) + ", error=" +
                std::to_string(error));
        }
    }
}

Tensor Pattern(const std::vector<int>& shape, float scale, float shift) {
    Tensor value(shape, 0.0f);
    for (size_t i = 0; i < value.Size(); ++i) {
        int bucket = static_cast<int>((i * 17 + 11) % 37);
        value.data[i] = static_cast<float>(bucket - 18) * scale + shift;
    }
    return value;
}

Tensor AddReference(const Tensor& a, const Tensor& b) {
    Require(a.shape == b.shape, "AddReference shape mismatch");
    Tensor result(a.shape, 0.0f);
    for (size_t i = 0; i < a.Size(); ++i) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

Tensor LinearReference(const Tensor& x, const Tensor& weight,
                       const Tensor* bias = nullptr) {
    bool one_dimensional = x.DimNum() == 1;
    int rows = one_dimensional ? 1 : x.shape[0];
    int in_features = one_dimensional ? x.shape[0] : x.shape[1];
    int out_features = weight.shape[0];
    Tensor result(one_dimensional ? std::vector<int>{out_features}
                                  : std::vector<int>{rows, out_features},
                  0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int out = 0; out < out_features; ++out) {
            float sum = bias == nullptr ? 0.0f : bias->data[out];
            for (int k = 0; k < in_features; ++k) {
                sum += x.data[static_cast<size_t>(row) * in_features + k] *
                       weight.data[static_cast<size_t>(out) * in_features + k];
            }
            result.data[static_cast<size_t>(row) * out_features + out] = sum;
        }
    }
    return result;
}

uint16_t FloatToHalfBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        uint32_t shifted = mantissa >> (1 - exponent);
        if ((shifted & 0x1000u) != 0) {
            shifted += 0x2000u;
        }
        return static_cast<uint16_t>(sign | (shifted >> 13));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if ((mantissa & 0x1000u) != 0) {
        mantissa += 0x2000u;
        if ((mantissa & 0x800000u) != 0) {
            mantissa = 0;
            if (++exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                 (mantissa >> 13));
}

float HalfBitsToFloat(uint16_t value) {
    uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t exponent32 = 127 - 15 + 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent32;
            }
            mantissa &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(exponent32) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<BlockQ41> QuantizeQ41ForTest(const Tensor& source) {
    int row_size = source.DimNum() >= 2 ? source.shape.back()
                                         : static_cast<int>(source.Size());
    int rows = source.DimNum() >= 2
                   ? static_cast<int>(source.Size()) / row_size
                   : 1;
    int blocks_per_row = (row_size + kQ41BlockSize - 1) / kQ41BlockSize;
    std::vector<BlockQ41> blocks;
    blocks.reserve(static_cast<size_t>(rows) * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
        for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
            int begin = row * row_size + block_index * kQ41BlockSize;
            int end = std::min(begin + kQ41BlockSize, (row + 1) * row_size);
            float minimum = std::numeric_limits<float>::infinity();
            float maximum = -std::numeric_limits<float>::infinity();
            for (int i = begin; i < end; ++i) {
                minimum = std::min(minimum, source.data[i]);
                maximum = std::max(maximum, source.data[i]);
            }
            BlockQ41 block{};
            float range = maximum - minimum;
            block.d = FloatToHalfBits(range > 0.0f ? range / 15.0f : 0.0f);
            block.m = FloatToHalfBits(std::isfinite(minimum) ? minimum : 0.0f);
            float d = HalfBitsToFloat(block.d);
            float m = HalfBitsToFloat(block.m);
            for (int i = 0; i < kQ41BlockSize / 2; ++i) {
                int first = begin + i;
                int second = first + kQ41BlockSize / 2;
                int q0 = first < end && d != 0.0f
                             ? static_cast<int>(std::round((source.data[first] - m) / d))
                             : 0;
                int q1 = second < end && d != 0.0f
                             ? static_cast<int>(std::round((source.data[second] - m) / d))
                             : 0;
                q0 = std::clamp(q0, 0, 15);
                q1 = std::clamp(q1, 0, 15);
                block.qs[i] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
            blocks.push_back(block);
        }
    }
    return blocks;
}

void ApplyRopeReference(Tensor& tensor, int pos, float theta, RopeType type) {
    int heads = tensor.shape[0];
    int head_dim = tensor.shape[1];
    int half = head_dim / 2;
    for (int head = 0; head < heads; ++head) {
        for (int pair = 0; pair < half; ++pair) {
            int first = type == RopeType::kNormal ? pair * 2 : pair;
            int second = type == RopeType::kNormal ? first + 1 : pair + half;
            float frequency = 1.0f /
                std::pow(theta, static_cast<float>(2 * pair) / head_dim);
            float cosine = std::cos(static_cast<float>(pos) * frequency);
            float sine = std::sin(static_cast<float>(pos) * frequency);
            size_t base = static_cast<size_t>(head) * head_dim;
            float x0 = tensor.data[base + first];
            float x1 = tensor.data[base + second];
            tensor.data[base + first] = x0 * cosine - x1 * sine;
            tensor.data[base + second] = x0 * sine + x1 * cosine;
        }
    }
}

Tensor AttentionReference(const Tensor& q, const KVCache& cache, int layer,
                          int pos, int n_heads, int n_kv_heads, int head_dim) {
    Tensor result({n_heads, head_dim}, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int heads_per_kv = n_heads / n_kv_heads;
    for (int head = 0; head < n_heads; ++head) {
        int kv_head = head / heads_per_kv;
        Tensor scores({pos + 1}, 0.0f);
        for (int token = 0; token <= pos; ++token) {
            const float* key = cache.GetKeyPtr(layer, token, kv_head);
            for (int dim = 0; dim < head_dim; ++dim) {
                scores.data[token] +=
                    q.data[static_cast<size_t>(head) * head_dim + dim] * key[dim];
            }
            scores.data[token] *= scale;
        }
        Tensor probabilities = Softmax(scores);
        for (int dim = 0; dim < head_dim; ++dim) {
            for (int token = 0; token <= pos; ++token) {
                result.data[static_cast<size_t>(head) * head_dim + dim] +=
                    probabilities.data[token] *
                    cache.GetValuePtr(layer, token, kv_head)[dim];
            }
        }
    }
    return result;
}

void TestRuntime(int device_id) {
    Require(CheckCudaRuntimeBuilt(), "CUDA runtime must be built");
    int count = GetCudaDeviceCount();
    Require(count > device_id, "selected CUDA device is unavailable");
    CudaDeviceInfo info = GetCudaDeviceInfo(device_id);
    Require(info.id == device_id && !info.name.empty() && info.compute_major > 0,
            "invalid CUDA device information");
    Require(info.total_memory_bytes > 0 && info.driver_version > 0 &&
                info.runtime_version > 0,
            "incomplete CUDA device information");

    std::vector<float> source = {1.0f, -2.5f, 3.25f, 7.0f};
    std::vector<float> output(source.size(), 0.0f);
    const size_t bytes = source.size() * sizeof(float);
    CudaBufferTracker tracker(true);
    CudaDeviceBuffer buffer(bytes, device_id);
    buffer.Upload(source.data(), bytes);
    buffer.Download(output.data(), bytes);
    Require(output == source, "CUDA upload/download round trip failed");
    CudaDeviceBuffer moved = std::move(buffer);
    Require(buffer.Empty() && !moved.Empty(), "CUDA buffer move failed");
    moved.Reset();
    Require(moved.Empty(), "CUDA buffer reset failed");
    CudaSynchronizeDevice(device_id);
    const auto stats = tracker.Snapshot();
    Require(stats.allocations == 1 && stats.peak_bytes == bytes && stats.live_bytes == 0,
            "CUDA buffer allocation stats mismatch after moving and resetting the buffer");
    Require(stats.h2d_copies == 1 && stats.d2h_copies == 1 &&
                stats.h2d_bytes == bytes && stats.d2h_bytes == bytes,
            "CUDA buffer transfer stats mismatch");
    std::cout << "PASS runtime: " << CudaFormatDeviceInfo(info) << "\n";
}

void TestMatmulAndOps(int device_id) {
    Require(CheckCudaMatmulBuilt(), "CUDA matmul must be built");
    Require(CheckCudaOpsBuild(), "CUDA ops must be built");
    Tensor a = Pattern({3, 17}, 0.031f, -0.07f);
    Tensor b = Pattern({17, 9}, 0.019f, 0.04f);
    RequireClose(CudaMatmul(a, b, device_id), Matmul(a, b), kFloatTolerance,
                 "CudaMatmul");

    Tensor x = Pattern({3, 33}, 0.027f, -0.09f);
    Tensor weight = Pattern({7, 33}, 0.017f, 0.05f);
    Tensor bias = Pattern({7}, 0.011f, -0.02f);
    RequireClose(CudaLinear(x, weight, &bias, device_id),
                 LinearReference(x, weight, &bias), kFloatTolerance,
                 "CudaLinear batch+bias");

    Tensor left = Pattern({3, 43}, 0.021f, -0.08f);
    Tensor right = Pattern({3, 43}, 0.013f, 0.12f);
    RequireClose(CudaElementMutmul(left, right, device_id),
                 ElementMutmul(left, right), kFloatTolerance,
                 "CudaElementMutmul non-block size");
    RequireClose(CudaElementAdd(left, right, device_id),
                 AddReference(left, right), kFloatTolerance,
                 "CudaElementAdd non-block size");
    RequireClose(CudaSilu(left, device_id), Silu(left), kFloatTolerance,
                 "CudaSilu non-block size");

    Tensor vector = Pattern({129}, 0.019f, -0.11f);
    Tensor norm_weight = Pattern({129}, 0.007f, 1.0f);
    RequireClose(CudaSoftmax(vector, device_id), Softmax(vector),
                 kFloatTolerance, "CudaSoftmax 129");
    RequireClose(CudaRmsNorm(vector, norm_weight, 1e-5f, device_id),
                 RmsNorm(vector, norm_weight, 1e-5f), kFloatTolerance,
                 "CudaRmsNorm 129");
    std::cout << "PASS matmul and elementwise ops\n";
}

void TestRope(int device_id) {
    for (RopeType type : {RopeType::kNormal, RopeType::kNeoX}) {
        Tensor q = Pattern({4, 14}, 0.023f, -0.05f);
        Tensor k = Pattern({2, 14}, 0.017f, 0.07f);
        Tensor expected_q = q;
        Tensor expected_k = k;
        ApplyRopeReference(expected_q, 11, 10000.0f, type);
        ApplyRopeReference(expected_k, 11, 10000.0f, type);
        CudaRope(q, k, 11, 10000.0f, type, device_id);
        std::string name = type == RopeType::kNormal ? "normal" : "neox";
        RequireClose(q, expected_q, kFloatTolerance, "CudaRope " + name + " q");
        RequireClose(k, expected_k, kFloatTolerance, "CudaRope " + name + " k");

        CudaTensor q_device = CudaTensorFromHost(expected_q, device_id);
        CudaTensor k_device = CudaTensorFromHost(expected_k, device_id);
        Tensor twice_q = expected_q;
        Tensor twice_k = expected_k;
        ApplyRopeReference(twice_q, 3, 10000.0f, type);
        ApplyRopeReference(twice_k, 3, 10000.0f, type);
        CudaRopeDeviceInput(q_device, k_device, 4, 2, 14, 3, 10000.0f,
                            type, device_id);
        RequireClose(q_device.Download(), twice_q, kFloatTolerance,
                     "CudaRopeDeviceInput " + name + " q");
        RequireClose(k_device.Download(), twice_k, kFloatTolerance,
                     "CudaRopeDeviceInput " + name + " k");
    }
    std::cout << "PASS normal and NeoX GQA RoPE\n";
}

void TestQuant(int device_id) {
    Require(CheckCudaQuantBuilt(), "CUDA quant kernels must be built");
    const std::vector<int> dimensions = {13, 32, 33, 64, 129};
    for (int in_features : dimensions) {
        int out_features = 5;
        Tensor weight = Pattern({out_features, in_features}, 0.019f, -0.04f);
        Tensor one_dim = Pattern({in_features}, 0.027f, 0.03f);
        Tensor batch = Pattern({3, in_features}, 0.023f, -0.02f);
        Tensor bias = Pattern({out_features}, 0.009f, 0.01f);

        std::vector<BlockQ80> q8 = QuantizeToQ80(weight);
        Tensor q8_expected = LinearQ80(batch, q8, weight.shape);
        RequireClose(CudaQ80Linear(one_dim, q8, weight.shape, &bias, device_id),
                     AddReference(LinearQ80(one_dim, q8, weight.shape), bias),
                     kQuantTolerance, "Q8_0 1D+bias dim=" + std::to_string(in_features));
        RequireClose(CudaQ80Linear(batch, q8, weight.shape, nullptr, device_id),
                     q8_expected, kQuantTolerance,
                     "Q8_0 batch dim=" + std::to_string(in_features));
        CudaDeviceBuffer q8_device(q8.size() * sizeof(BlockQ80), device_id);
        q8_device.Upload(q8.data(), q8.size() * sizeof(BlockQ80));
        CudaTensor batch_device = CudaTensorFromHost(batch, device_id);
        RequireClose(CudaQ80LinearDeviceInput(batch_device, q8_device.GetData(),
                                              q8.size(), weight.shape, device_id)
                         .Download(),
                     q8_expected, kQuantTolerance,
                     "Q8_0 device batch dim=" + std::to_string(in_features));

        std::vector<BlockQ40> q4 = QuantizeToQ40(weight);
        Tensor q4_expected = LinearQ40(batch, q4, weight.shape);
        RequireClose(CudaQ40Linear(one_dim, q4, weight.shape, &bias, device_id),
                     AddReference(LinearQ40(one_dim, q4, weight.shape), bias),
                     kQuantTolerance, "Q4_0 1D+bias dim=" + std::to_string(in_features));
        RequireClose(CudaQ40Linear(batch, q4, weight.shape, nullptr, device_id),
                     q4_expected, kQuantTolerance,
                     "Q4_0 batch dim=" + std::to_string(in_features));
        CudaDeviceBuffer q4_device(q4.size() * sizeof(BlockQ40), device_id);
        q4_device.Upload(q4.data(), q4.size() * sizeof(BlockQ40));
        RequireClose(CudaQ40LinearDeviceInput(batch_device, q4_device.GetData(),
                                              q4.size(), weight.shape, device_id)
                         .Download(),
                     q4_expected, kQuantTolerance,
                     "Q4_0 device batch dim=" + std::to_string(in_features));

        std::vector<BlockQ41> q41 = QuantizeQ41ForTest(weight);
        Tensor q41_expected = LinearQ41(batch, q41, weight.shape);
        RequireClose(CudaQ41Linear(one_dim, q41, weight.shape, &bias, device_id),
                     AddReference(LinearQ41(one_dim, q41, weight.shape), bias),
                     kQuantTolerance, "Q4_1 1D+bias dim=" + std::to_string(in_features));
        RequireClose(CudaQ41Linear(batch, q41, weight.shape, nullptr, device_id),
                     q41_expected, kQuantTolerance,
                     "Q4_1 batch dim=" + std::to_string(in_features));
        CudaDeviceBuffer q41_device(q41.size() * sizeof(BlockQ41), device_id);
        q41_device.Upload(q41.data(), q41.size() * sizeof(BlockQ41));
        RequireClose(CudaQ41LinearDeviceInput(batch_device, q41_device.GetData(),
                                              q41.size(), weight.shape, device_id)
                         .Download(),
                     q41_expected, kQuantTolerance,
                     "Q4_1 device batch dim=" + std::to_string(in_features));
    }
    std::cout << "PASS Q8_0, Q4_0 and Q4_1 linear boundary cases\n";
}

void TestCacheAndAttention(int device_id, int target_pos, int head_dim) {
    Require(CheckCudaKVCacheBuilt(), "CUDA KV cache must be built");
    Require(CheckCudaAttentionBuilt(), "CUDA attention must be built");
    constexpr int layers = 2;
    const int max_seq_len = target_pos + 7;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int target_layer = 1;
    KVCache cpu_cache(layers, max_seq_len, n_kv_heads, head_dim);
    CudaKVCache cuda_cache(layers, max_seq_len, n_kv_heads, head_dim, device_id);
    for (int pos = 0; pos <= target_pos; ++pos) {
        Tensor key = Pattern({n_kv_heads, head_dim}, 0.013f * (pos + 1), -0.04f);
        Tensor value = Pattern({n_kv_heads, head_dim}, 0.017f * (pos + 1), 0.03f);
        cpu_cache.Write(target_layer, pos, key, value);
        if (pos == target_pos) {
            CudaTensor key_device = CudaTensorFromHost(key, device_id);
            CudaTensor value_device = CudaTensorFromHost(value, device_id);
            cuda_cache.WriteDevice(target_layer, pos, key_device, value_device);
        } else {
            cuda_cache.Write(target_layer, pos, key, value);
        }
        RequireClose(cuda_cache.ReadKey(target_layer, pos), key, 0.0f,
                     "CUDA KV key round trip");
        RequireClose(cuda_cache.ReadValue(target_layer, pos), value, 0.0f,
                     "CUDA KV value round trip");
    }
    Tensor q = Pattern({n_heads, head_dim}, 0.021f, -0.06f);
    Tensor expected = AttentionReference(q, cpu_cache, target_layer, target_pos,
                                         n_heads, n_kv_heads, head_dim);
    Tensor actual = CudaAttentionDecode(q, cuda_cache, target_layer, target_pos,
                                        max_seq_len, n_heads, n_kv_heads,
                                        head_dim, device_id);
    RequireClose(actual, expected, kQuantTolerance, "CUDA GQA attention");
    CudaTensor q_device = CudaTensorFromHost(q, device_id);
    RequireClose(CudaAttentionDecodeDeviceInputs(
                     q_device, cuda_cache, target_layer, target_pos, max_seq_len,
                     n_heads, n_kv_heads, head_dim, device_id)
                     .Download().ReshapeAndCheck({n_heads, head_dim},
                                                "device attention output"),
                 expected, kQuantTolerance, "CUDA device-input GQA attention");
    cuda_cache.Clear();
    RequireClose(cuda_cache.ReadKey(target_layer, target_pos),
                 Tensor({n_kv_heads, head_dim}, 0.0f), 0.0f,
                 "CUDA KV clear");
    std::cout << "PASS CUDA KV cache and GQA attention: pos=" << target_pos
              << ", head_dim=" << head_dim << "\n";
}

void TestCudaArgMax(int device) {
    CudaDeviceBuffer workspace;
    Tensor logits({1031}, -std::numeric_limits<float>::max());
    CudaTensor device_logits = CudaTensorFromHost(logits, device);
    Require(CudaArgMax(device_logits, workspace) == 0, "CUDA ArgMax all tied");
    logits.data[256] = 5.0f;
    logits.data[1030] = 5.0f;
    device_logits.Upload(logits);
    Require(CudaArgMax(device_logits, workspace) == 256, "CUDA ArgMax cross-block tie");
    logits.data[1030] = 6.0f;
    device_logits.Upload(logits);
    Require(CudaArgMax(device_logits, workspace) == 1030, "CUDA ArgMax last element");
    for (const float value : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
        logits.data[5] = value;
        device_logits.Upload(logits);
        bool rejected = false;
        try {
            CudaArgMax(device_logits, workspace);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("not finite") != std::string::npos;
        }
        Require(rejected, "CUDA ArgMax accepted a non-finite logit");
    }
    bool rejected = false;
    try {
        CudaArgMax(CudaTensor{}, workspace);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("non-empty") != std::string::npos;
    }
    Require(rejected, "CUDA ArgMax accepted an empty logit vector");
    std::cout << "PASS CUDA ArgMax\n";
}

Tensor CopyRows(const Tensor& matrix, int offset, int count, std::vector<int> shape) {
    Tensor rows(shape, 0.0f);
    std::copy_n(matrix.data.begin() + static_cast<size_t>(offset) * matrix.shape[1],
                rows.Size(), rows.data.begin());
    Require(rows.Size() == static_cast<size_t>(count) * matrix.shape[1], "row shape mismatch");
    return rows;
}

void TestPrefillAttention(int device) {
    constexpr int heads = 14;
    constexpr int kv_heads = 2;
    constexpr int dim = 64;
    constexpr int first = 13;
    constexpr int rows = 19;
    constexpr int length = first + rows + 3;
    const Tensor keys = Pattern({length, kv_heads * dim}, 0.15f, -0.08f);
    const Tensor values = Pattern({length, kv_heads * dim}, 0.04f, 0.05f);
    const Tensor queries = Pattern({rows, heads * dim}, 1.1f, 0.3f);
    KVCache cpu_cache(2, length, kv_heads, dim);
    CudaKVCache gpu_cache(2, length, kv_heads, dim, device);
    for (int pos = 0; pos < length; ++pos) {
        cpu_cache.Write(1, pos, CopyRows(keys, pos, 1, {kv_heads, dim}),
                        CopyRows(values, pos, 1, {kv_heads, dim}));
    }
    const auto write_gpu_rows = [&](int pos, int count) {
        auto k = CudaTensorFromHost(CopyRows(keys, pos, count, {count, kv_heads * dim}), device);
        auto v = CudaTensorFromHost(CopyRows(values, pos, count, {count, kv_heads * dim}), device);
        gpu_cache.WriteDeviceBatch(1, pos, k, v);
    };
    write_gpu_rows(0, first);
    write_gpu_rows(first, length - first);
    const Tensor actual = CudaAttentionPrefill(CudaTensorFromHost(queries, device),
                                               gpu_cache, 1, first, heads).Download();
    for (int row = 0; row < rows; ++row) {
        const auto query = CopyRows(queries, row, 1, {heads, dim});
        const auto expected = AttentionReference(query, cpu_cache, 1, first + row, heads, kv_heads, dim);
        RequireClose(CopyRows(actual, row, 1, {heads, dim}), expected, kQuantTolerance,
                     "causal GQA prefill with prefix and future slots");
    }
    Require(!CudaPrefillAttentionSupports(14, 2, 128) && !CudaPrefillAttentionSupports(18, 2, dim),
            "unsupported CUDA prefill shape accepted");
    std::cout << "PASS tiled causal GQA Prefill\n";
}

void TestForwardPrefill(int device) {
    auto model = LoadModel("models/tiny/model.json", "models/tiny/model.bin");
    Require(model.is_loaded, "tiny CUDA Prefill model load failed");
    model.config.max_seq_len = 160;
    model.config.rope_type = RopeType::kNeoX;
    const int q_size = model.config.n_heads * model.config.head_dim;
    const int kv_size = model.config.n_kv_heads * model.config.head_dim;
    model.layers[0].bq = Pattern({q_size}, 0.001f, 0.0f);
    model.layers[0].bk = Pattern({kv_size}, 0.002f, -0.01f);
    model.layers[0].bv = Pattern({kv_size}, 0.001f, 0.02f);
    UploadModelWeightsToCuda(model, device);
    model.cuda_weights->batch_prefill_enabled = true;
    MiniLlamaContext sequential(&model);
    MiniLlamaContext batched(&model);
    std::vector<int> tokens(153);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = 32 + static_cast<int>(i) % 90;
    }
    Tensor expected;
    for (int pos = 0; pos < 23; ++pos) {
        expected = ForwardBatch(sequential, model, MiniBatch::GetSingleMiniBatch(tokens[pos], pos));
    }
    ForwardBatch(batched, model, MiniBatch::GetMultiMiniBatch(std::vector<int>(tokens.begin(), tokens.begin() + 5)));
    Require(batched.last_batch_used_cuda_prefill, "initial CUDA Prefill did not batch");
    auto actual = ForwardBatch(batched, model,
        MiniBatch::GetMultiMiniBatch(std::vector<int>(tokens.begin() + 5, tokens.begin() + 23), 5));
    Require(batched.last_batch_used_cuda_prefill && batched.pos == 22 && batched.token_history.size() == 23,
            "CUDA Prefill did not commit the expected token prefix");
    RequireClose(actual, expected, kQuantTolerance, "CUDA Prefill final logits");
    for (int layer = 0; layer < model.config.n_layers; ++layer) {
        for (int pos : {0, 4, 5, 22}) {
            RequireClose(batched.cuda_kv_cache_.ReadKey(layer, pos),
                         sequential.cuda_kv_cache_.ReadKey(layer, pos), kQuantTolerance, "batched key");
            RequireClose(batched.cuda_kv_cache_.ReadValue(layer, pos),
                         sequential.cuda_kv_cache_.ReadValue(layer, pos), kQuantTolerance, "batched value");
        }
    }
    const auto decode = MiniBatch::GetSingleMiniBatch(tokens[23], 23);
    RequireClose(ForwardBatch(batched, model, decode), ForwardBatch(sequential, model, decode),
                 kQuantTolerance, "Decode after CUDA Prefill");
    Require(!batched.last_batch_used_cuda_prefill, "single token did not fall back");

    auto chunks = ForwardBatch(batched, model, MiniBatch::GetMultiMiniBatch(std::vector<int>(tokens.begin() + 24, tokens.end()), 24));
    Require(batched.last_batch_used_cuda_prefill && batched.token_history == tokens,
            "chunked CUDA Prefill did not commit the whole suffix");
    ForwardBatch(sequential, model, MiniBatch::GetMultiMiniBatch(std::vector<int>(tokens.begin() + 24, tokens.end() - 1), 24));
    RequireClose(chunks, ForwardBatch(sequential, model, MiniBatch::GetSingleMiniBatch(tokens.back(), 152)),
                 kQuantTolerance, "Prefill across a 128-token chunk boundary");

    for (const auto& bad : {
            MiniBatch::GetMultiMiniBatch({1, model.config.vocab_size}, 153),
            MiniBatch::GetMultiMiniBatch(std::vector<int>(8, 1), 153)}) {
        bool rejected = false;
        try {
            ForwardBatch(batched, model, bad);
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        Require(rejected && batched.pos == 152 && batched.token_history == tokens,
                "invalid batch advanced the CUDA prefix");
    }
    MiniLlamaContext noncontiguous(&model);
    ForwardBatch(noncontiguous, model, MiniBatch{{1, 2}, {0, 2}});
    Require(!noncontiguous.last_batch_used_cuda_prefill && noncontiguous.cuda_valid_prefix == 1,
            "noncontiguous CUDA batch should use the compatibility path");

    const auto norm = std::find_if(model.cuda_weights->weights.begin(), model.cuda_weights->weights.end(),
                                   [](const auto& weight) { return weight.name == "layers.1.attention_norm"; });
    Require(norm != model.cuda_weights->weights.end(), "test norm weight unavailable");
    ++norm->shape[0];
    bool failed = false;
    try {
        ForwardBatch(batched, model, MiniBatch::GetMultiMiniBatch({1, 2}, 153));
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("weight shape") != std::string::npos;
    }
    Require(failed && batched.token_history.empty() && batched.cuda_kv_cache_.Empty() &&
                batched.cuda_valid_prefix == 0,
            "failed CUDA Prefill left a reusable partially-written prefix");
    std::cout << "PASS CUDA Prefill, KV parity, chunking and cache invalidation\n";
}

}  // namespace
}  // namespace mini_llama

int main(int argc, char** argv) {
    try {
        int device_id = 0;
        if (argc == 3 && std::string(argv[1]) == "--device") {
            device_id = std::stoi(argv[2]);
        } else if (argc != 1) {
            throw std::runtime_error("usage: gpu-core-test [--device N]");
        }
        mini_llama::TestRuntime(device_id);
        mini_llama::TestMatmulAndOps(device_id);
        mini_llama::TestRope(device_id);
        mini_llama::TestQuant(device_id);
        mini_llama::TestCudaArgMax(device_id);
        mini_llama::TestPrefillAttention(device_id);
        mini_llama::TestForwardPrefill(device_id);
        for (int head_dim : {7, 128, 129}) {
            for (int pos : {0, 4, 126, 127, 128, 256}) {
                mini_llama::TestCacheAndAttention(device_id, pos, head_dim);
            }
        }
        std::cout << "PASS gpu-core-test\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL gpu-core-test: " << error.what() << "\n";
        return 1;
    }
}
