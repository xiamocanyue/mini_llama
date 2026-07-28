#pragma once 

#include "include/tensor.h"

namespace mini_llama {

enum class MatmulMode {
    kNaive,
    kThread,
    kSimd,
    kThreadAndSimd,
};

MatmulMode DefalutMatulMode();

// 简单矩阵乘法接口
Tensor MatulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// x * w_d 接口
Tensor LinearDispatch(const Tensor& x, const Tensor& w, MatmulMode model);

}