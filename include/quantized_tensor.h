#pragma once 

#include "include/tensor.h"

namespace mini_llama {

enum class QuantType {
    kF32,
    kQ80,
    kQ40,
    kQ41,
};

// 每个块有32个元素
constexpr int kQ80BlockSize = 32;
constexpr int kQ40BlockSize = 32;
constexpr int kQ41BlockSize = 32;

struct BlockQ80 {
    u_int16_t d;       
    int8_t qs[32];
};
static_assert(sizeof(BlockQ80) == sizeof(u_int16_t) + kQ80BlockSize, "Q80 block size is wrong");

struct BlockQ40 {
    u_int16_t d;
    u_int8_t qs[16];
};
static_assert(sizeof(BlockQ40) == sizeof(u_int16_t) + kQ40BlockSize / 2, "Q40 block size is wrong");

//有偏移的q40量化
struct BlockQ41 {
    u_int16_t d;
    u_int16_t m;
    u_int8_t qs[16];
};
static_assert(sizeof(BlockQ41) == sizeof(u_int16_t) * 2 + kQ41BlockSize / 2, "Q41 block size is wrong");


struct  QuantizedTensor {
    QuantizedTensor(QuantType t, const std::vector<int>& sh) : type(t), shape(sh) {}
    QuantizedTensor() = default;

    size_t ElementsNum() const {
        size_t n = 1;
        for (int s : shape) {
            n *= static_cast<size_t>(s);
        }
        return n;
    }

    int DimSum() const {
        return static_cast<int>(shape.size());
    }

    std::string ShapeString() const {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); i++) {
            s += std::to_string(shape[i]);
            if (i + 1 < shape.size()) {
                s += ',';
            }
        }
        s += ']';
        return s;
    }

    void AssertShape(const std::vector<int>& expect, const char* caller) const {
        if (shape != expect) {
            throw std::runtime_error(std::string(caller) + " shape mismatch, expect " + 
                                    QuantizedTensor{QuantType::kF32, expect}.ShapeString() + 
                                    ". got " + ShapeString());
        }
    }

    QuantType type = QuantType::kF32;
    std::vector<int> shape;
    std::vector<float> f32_data;
    std::vector<BlockQ80> q8_0_data;
    std::vector<BlockQ40> q4_0_data;
    std::vector<BlockQ41> q4_1_data;
};
//将量化算子变成普通算子
Tensor ToTensor(const QuantizedTensor& qtensor);
//将普通算子变成量化算子
QuantizedTensor ToQuantizedTensor(const Tensor& tensor);

}