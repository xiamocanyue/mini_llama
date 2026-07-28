#include "include/matmul_dispatch.h"


#include <vector>

#include "include/thread_pool.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

namespace mini_llama {

//[m, o] * [o, n] -> [m, n]
static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
    int m = a.shape[0];
    int o = a.shape[1];
    int n = b.shape[1];
    Tensor c({m, n}, 0.0f);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < o; k++) {
                sum += a.At2(i, k) * b.At2(k, j);
            }
            c.At2(i, j) = sum;
        }
    }
    return c;
}

static Tensor MatmulThread(const Tensor&  a, const Tensor& b) {
    int m = a.shape[0];
    int o = a.shape[1];
    int n = b.shape[1];
    Tensor c({m, n}, 0.0f);
    ParallelFor(m, [&](int begin, int end) {
        for (int i = begin; i < end; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < o; k++) {
                    sum += a.At2(i, k) * b.At2(k, j);
                }
                c.At2(i, j) = sum;
            }
        }
    });
    return c;
}




//x 规定是一行的向量，或者只有一行的二维向量,即一个token
//x [in_feature]
//w [out_feature, in_feature] , 取得时候按行取，相当于转置的按列取
//x * w^T  -> [1, out_feature]
static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
    int in_feature = weight.shape[1];
    int out_feature = weight.shape[0];
    bool is_1d = (x.DimNum() == 1);
    Tensor result(is_1d ? std::vector<int>{out_feature} : std::vector<int>{1, out_feature}, 0.0f);
    for (int i = 0; i < out_feature; i++) {
        float sum = 0.0f;
        for (int k = 0; k < in_feature; k++) {
            sum += x.At1(k) * weight.At2(i, k);
        
        }
        result.At1(i) = sum;
    }
    return result;
}

static Tensor LinearThread(const Tensor& x, const Tensor& weight) {
    int in_feature = weight.shape[1];
    int out_feature = weight.shape[0];
    bool is_1d = (x.DimNum() == 1);
    Tensor result(is_1d ? std::vector<int>{out_feature} : std::vector<int>{1, out_feature}, 0.0f);
    ParallelFor(out_feature, [&](int begin, int end) {
        for (int i = begin; i < end; i++) {
            float sum = 0.0f;
            for (int k = 0; k < in_feature; k++) {
                sum += x.At1(k) * weight.At2(i, k);
            }
            result.At1(i) = sum;
        }
    });
    return result;
}


#ifdef MINI_LLAMA_USE_NEON
static float DotSimdNeon(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float32x2_t r = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    r = vpadd_f32(r, r);
    float result = vget_lane_f32(r, 0);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

#ifdef MINI_LLAMA_USE_AVX2
static float DotSimdAvx2(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // Horizontal reduce
    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    sum_lo = _mm_add_ps(sum_lo, sum_hi);
    sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
    sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
    float result = _mm_cvtss_f32(sum_lo);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

static float DotSimd(const float* a, const float* b, int n) {
#if defined(MINI_LLAMA_USE_AVX2)
    return DotSimdAvx2(a, b, n);
#elif defined(MINI_LLAMA_USE_NEON)
    return DotSimdNeon(a, b, n);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

static Tensor LinearSimd(const Tensor& x, const Tensor& weight) {
    int in_feature = weight.shape[1];
    int out_feature = weight.shape[0];
    bool is_1d = (x.DimNum() == 1);
    Tensor result(is_1d ? std::vector<int>{out_feature} : std::vector<int>{1, out_feature}, 0.0f);
    for (int i = 0; i < out_feature; i++) {
        result.At1(i) = DotSimd(x.data.data(), weight.data.data() + i * in_feature, in_feature);
    }
    return result;
}

static Tensor LinearThreadAndSimd(const Tensor& x, const Tensor& weight) {
    int in_feature = weight.shape[1];
    int out_feature = weight.shape[0];
    bool is_1d = (x.DimNum() == 1);
    Tensor result(is_1d ? std::vector<int>{out_feature} : std::vector<int>{1, out_feature}, 0.0f);
    ParallelFor(out_feature, [&](int begin, int end) {
        for (int i = begin; i < end; i++) {
            result.data[i] = DotSimd(x.data.data(), weight.data.data() + i * in_feature, in_feature);
        }
    });
    return result;
}



MatmulMode DefalutMatulMode() {
    return MatmulMode::kThreadAndSimd;
}

// 简单矩阵乘法接口
Tensor MatulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
    if (a.DimNum() != 2 || b.DimNum() != 2) {
        throw std::runtime_error("MatmulDispatch : expect 2D tensors");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("MatmulDispatch : dimension mismatch");
    }
    switch (mode) {
        case MatmulMode::kNaive:
            return MatmulNaive(a, b);
        case MatmulMode::kThread:
            return MatmulThread(a, b);
        case MatmulMode::kSimd:
        case MatmulMode::kThreadAndSimd:
            return MatmulThread(a, b);
        default:
            return MatmulThread(a, b);
    }
    return MatmulThread(a, b);
}

// x * w_d 接口
Tensor LinearDispatch(const Tensor& x, const Tensor& w, MatmulMode model) {
    if (w.DimNum() != 2) {
        throw std::runtime_error("LinearDispatch: weight shape is not [out_feature, in_feature]");
    }
    int in_feature;
    if (x.DimNum() == 1) {
        in_feature = x.shape[0];
    } else if (x.DimNum() == 2 && x.shape[0] == 1) {
        in_feature = x.shape[1];
    } else {
        throw std::runtime_error("LinearDispatch: x shape is not [in_feature] or [1, in_feature]");
    }
    if (w.shape[1] != in_feature) {
        throw std::runtime_error("LinearDispatch: weight is not match x");
    }

    switch (model) {
        case MatmulMode::kNaive:
            return LinearNaive(x, w);
        case MatmulMode::kSimd:
            return LinearSimd(x, w);
        case MatmulMode::kThread:
            return LinearThread(x, w);
        case MatmulMode::kThreadAndSimd:
            return LinearThreadAndSimd(x, w);
    }
    return LinearThreadAndSimd(x, w);
}




}