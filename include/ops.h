#pragma once 

#include "include/tensor.h"
#include "include/quantized_tensor.h"
#include "include/model.h"

namespace mini_llama  {

// 矩阵乘法 返回一个c矩阵， c = a * b
// [i, j] * [j, k] -> [i, k]
Tensor Matmul(const Tensor& a, const Tensor& b);

//逐个元素相乘
Tensor ElementMutmul(const Tensor& a, const Tensor& b);

//计算 x * w_d   
Tensor Linear(const Tensor& x, const Tensor& weight);

//计算 x * w_d（量化后的权重矩阵）
Tensor Linear(const Tensor& x, const QuantizedTensor& weight);

// y = x / sqrt(mean(x^2) + eps) * weight
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

//e^(x - max) / sum(e^(x - max))
Tensor Softmax(const Tensor& x);

// x * sigmoid(x)
Tensor Silu(const Tensor& x);

//Swiglu = Silu(gate) · up
Tensor Swiglu(const Tensor& gate, const Tensor& up);

//x1` = x1 * cos(m * theta) - x2 * sin(m * theta)
//x2` = x1 * sin(m * theta) + x2 * cos(m * theta)
//void RoPE(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type = RopeType::kNormal);
void RoPE(Tensor& q, Tensor& k, int pos, const std::vector<float>& cos_table,
          const std::vector<float>& sin_table, RopeType rope_type = RopeType::kNormal);

//返回数组最大值
int GetArgMax(const Tensor& x);
    
} // namespace mini_llama 
