#pragma once 

#include <vector>
#include "include/quantized_tensor.h"

namespace mini_llama {

std::vector<BlockQ80> QuantizeToQ80(const Tensor& t);
Tensor DeQuantizeFromQ80(const std::vector<BlockQ80>& blocks, const std::vector<int>& shape);

std::vector<BlockQ40> QuantizeToQ40(const Tensor& t);
Tensor DeQuantizeFromQ40(const std::vector<BlockQ40>& blocks, const std::vector<int>& shape);

Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks, const std::vector<int>& shape);

Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight, const std::vector<int>& weight_shape);
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight, const std::vector<int>& weight_shape);
Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight, const std::vector<int>& weight_shape);

float CompareMatmulError(const Tensor& w, const Tensor& input);
float CompareQ40Error(const Tensor& w, const Tensor& input);


}