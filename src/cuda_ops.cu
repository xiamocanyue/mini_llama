#include "include/cuda_ops.h"

#include <cuda_runtime_api.h>


#include "include/cuda_data.h"

namespace mini_llama {
namespace {
constexpr int kBlockSize = 256;

void CheckCudaOps(cudaError_t err, const char* caller) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA ops error from " + std::string(caller) + ": " + cudaGetErrorString(err));
    }
}

void CheckLastKernelError(const char* caller) {
    CheckCudaOps(cudaGetLastError(), caller);
}

__global__ void SumSquareKernel(const float* x, float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(sum, static_cast<float>(x[i] * x[i]));
    }
}

// y = x / sqrt(mean(x^2) + eps) * weight
__global__ void RmsNormKernel(const float* x, const float* weight, float* y, const float* sum, int n, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float scale = rsqrt((*sum / static_cast<float>(n)) + eps);
        y[i] = x[i] * scale * weight[i];
    }
}

// y = x / (1 + exp(-x))
__global__ void SiluKernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] / (1.0f + expf(x[i] * -1));
    }
}

__global__ void ElementMulmutKernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] * b[i];
     }
}

__global__ void ElementAddKernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
     }
}

//可以用__shfl_down_sync优化
__global__ void GetMaxForSoftmaxKernel(const float* x, float* max_out, int n) {
    __shared__ float shared[kBlockSize];
    float  max = -3.402823466e+38F;
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        max = fmaxf(max, x[i]);
    }
    shared[tid] = max;
    __syncthreads;

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads;
    }
    if (tid == 0) {
        *max_out = shared[0];
    }
}

//e^(x - max) / sum(e^(x - max))
// y = e^(x - max);
__global__ void GetExpForSoftmaxKernel(const float* x, float* y, const float* max, float* sum_out, int n) {
    __shared__ float shared[kBlockSize];
    float local_sum = 0.0f;
    float max_v = *max; 
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        float e = expf(x[i] - max_v);
        y[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    __syncthreads;

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];            
        }
        __syncthreads;
    }
    if (tid == 0) {
        *sum_out = shared[0];
    }
}

//e^(x - max) / sum(e^(x - max))
__global__ void SoftmaxKernel(float* y, const float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] /= *sum; 
    }
}

//x1` = x1 * cos(m * freq) - x2 * sin(m * freq)
//x2` = x1 * sin(m * freq) + x2 * cos(m * freq)
//freq = 1 / theta^(2 * i / head_dim)  i是pair的下标, 2*i就是元素的下标
//pair <x, y>，一个线程处理两个相邻的元素
//pair_index 是处理pair的线程下标
__global__ void RopeNormalKernel(float* x, int n_heads, int head_dim, int pos, float theta) {
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    int head_have_pairs = head_dim / 2;
    int total_pairs = n_heads * head_have_pairs;
    if (pair_index > total_pairs) {
        return ;
    }

    int head_index = pair_index / head_have_pairs;
    int pair_index = pair_index % head_have_pairs;
    int num_index = pair_index * 2;
    int base = head_index * head_dim + num_index;

    float freq = 1.0f / powf(theta, static_cast<float>(num_index) / static_cast<float>(head_dim));
    float cos_val = cosf(static_cast<float>(pos) * freq);
    float sin_val = sinf(static_cast<float>(pos) * freq);
    float x1 = x[base];
    float x2 = x[base + 1];

    x[base] = x1 * cos_val - x2 * sin_val;
    x[base + 1] = x1 * sin_val + x2 * cos_val;
}

__global__ void RopeNeoXKernel(float* x, int n_heads, int head_dim, int pos, float theta) {
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    int head_have_pairs = head_dim / 2;
    int half_dim = head_have_pairs;
    int total_pairs = n_heads * head_have_pairs;
    if (pair_index > total_pairs) {
        return ;
    }

    int head_index = pair_index / head_have_pairs;
    int pair_index = pair_index % head_have_pairs;
    int base = head_index * head_dim;

    float freq = 1.0f / powf(theta, static_cast<float>(2 * pair_index) / static_cast<float>(head_dim));
    float cos_val = cosf(static_cast<float>(pos) * freq);
    float sin_val = sinf(static_cast<float>(pos) * freq);
    float x1 = x[base + pair_index];
    float x2 = x[base + pair_index + half_dim];

    x[base + pair_index] = x1 * cos_val - x2 * sin_val;
    x[base + pair_index + half_dim] = x1 * sin_val + x2 * cos_val;
}

__global__ void EmbeddingLookUpKernel(const float* embedding, float* y, int token_id, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        y[i] = embedding[token_id * dim + i];
    }
}

void RequireSameShape(const Tensor& a, const Tensor& b, const char* caller) {
    if (a.shape != b.shape) {
        throw std::runtime_error(std::string(caller) + ": shape is not match form " + a.ShapeString() + " to " + b.ShapeString());
    }
}

void RequireSameShape(const CudaTensor& a, const CudaTensor& b, const char* caller) {
    if (a.Shape() != b.Shape()) {
        throw std::runtime_error(std::string(caller) + ": shape is not match form " + a.ShapeString() + " to " + b.ShapeString());
    } 
    if (a.GetDeviceId() != b.GetDeviceId()) {
        throw std::runtime_error(std::string(caller) + "tensor is on differen cuda devices");
    }
}

void Check1D(const Tensor& x, const char* caller) {
    if (x.DimNum() != 1) {
        throw std::runtime_error(std::string(caller) + ": tensor shape is not 1D, is " + x.ShapeString());
    }
    if (x.Size() == 0) {
        throw std::runtime_error(std::string(caller) + ": tensor is empty");
    }
}

void Check1D(const CudaTensor& x, const char* caller) {
    if (x.GetDimNum() != 1) {
        throw std::runtime_error(std::string(caller) + ": tensor shape is not 1D, is " + x.ShapeString());
    }
    if (x.GetSize() == 0) {
        throw std::runtime_error(std::string(caller) + "tensor is empty");
    }
}

void ValidateRopeInput(const Tensor& q, const Tensor& k, int pos, float theta) {
    if (q.DimNum() != 2 || k.DimNum() != 2) {
        throw std::runtime_error("CudaRope: tensor is not 2D");
    }
    if (pos < 0) {
        throw std::out_of_range("CudaRope: pos must be positive");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("CudaRope: theta must be finite and positive");
    }
    if (q.shape[1] != k.shape[1]) {
        throw std::runtime_error("CudaRope: q and k head_dim is not match");
    }
    if (q.shape[1] <= 0 || q.shape[1] % 2 != 0) {
        throw std::runtime_error("CudaRope: head_dim must be even and positive");
    }
}

void ValidateRopeDeviceInput(const CudaTensor& q, const CudaTensor& k, int pos, float theta, int n_heads, int n_kv_heads, int head_dim, int device_id) {
    if (q.GetDeviceId() != device_id || k.GetDeviceId() != device_id) {
        throw std::runtime_error("CudaRopeDeviceInput: input tensor is not on this CUDA device");
    }
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error("CudaRopeDeviceInput: head shape is invalid");
    } 
    if (q.GetSize() != static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
        throw std::runtime_error("CudaRopeDeviceInput: q shape is not match");
    }
    if (k.GetSize() != static_cast<size_t>(n_kv_heads) * static_cast<int>(head_dim)) {
        throw std::runtime_error("CudaRopeDeviceInput: k shape is not match");
    }
    if (pos < 0) {
        throw std::out_of_range("CudaRopeDeviceInput: pos must be positive");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("CudaRopeDeviceInput: theta must be positive");
    }
}

void ValidateEmbeddingDeviceWeightInput(const void* embedding_data, const std::vector<int>& embedding_shape, int token_id) {
    if (embedding_data == nullptr) {
        throw std::runtime_error("CudaEmbeddingDevcieWeightInput: embedding is empty");
    }
    if (embedding_shape.size() != 2) {
        throw std::runtime_error("CudaEmbeddingDevcieWeightInput: shape must be 2D");
    }

    int vocab_size = embedding_shape[0];
    int dim = embedding_shape[1];
    if (vocab_size <= 0 || dim <= 0) {
        throw std::runtime_error("CudaEmbeddingDevcieWeightInput: embedding shape must be positive");
    }
    if (token_id < 0 || token_id >= vocab_size) {
        throw std::runtime_error("CudaEmbeddingDevcieWeightInput: token is out of range");
    }
}

void ValidateRmsNormInput(const Tensor& x, const Tensor& weight, float eps, int device_id) {
    Check1D(x, "CudaRmsNorm");
    Check1D(weight, "CudaRmsNorm");
    if (x.shape != weight.shape) {
        throw std::runtime_error("CudaRmsNorm: x and weight shape is not match");
    }
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("CudaRmsNorm: eps must be positive and finite");
    }
}

void ValidateRmsNormDeviceWeightInput(const CudaTensor& x, const void* weight, const std::vector<int>& weight_shape, float eps, int device_id) {
    Check1D(x, "CudaRmsNormDeviceWeightInput");
    if (weight == nullptr) {
        throw std::runtime_error("CudaRmsNormDeviceWeightInput: weight is empty");
    }
    if (x.Shape() != weight_shape) {
        throw std::runtime_error("CudaRmsNormDeviceWeightInput: shape is not match");
    }
    if (x.GetDeviceId() != device_id) {
        throw std::runtime_error("CudaRmsNormDeviceWeightInput: current tensor is not on this device");
    }
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("CudaRmsNormDeviceWeightInput: eps must be positive and finite");
    }
}

void ValidateAddDeviceWeightInput(const CudaTensor& a, const void* data, const std::vector<int>& data_shape, int device_id) {
    if (a.GetDeviceId() != device_id) {
        throw std::runtime_error("CudaAddDeviceWeightInput: current tensor is not on this device");
    }
    if (data == nullptr) {
        throw std::runtime_error("CudaAddDeviceWeightInput: data is empty");
    }
    if (a.Shape() != data_shape) {
        throw std::runtime_error("CudaAddDeviceWeightInput: input and weight is not match");
    }
}

int GetBlockNum(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

} //namespace


Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps, int device_id) {
    ValidateRmsNormInput(x, weight, eps, device_id);

    CudaSetDevice(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.Size() * sizeof(float), device_id);
    CudaDeviceBuffer w_dev(weight.Size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
    CudaDeviceBuffer sum_dev(sizeof(float), device_id);

    x_dev.Upload(x.data.data(), x.Size() * sizeof(float));
    w_dev.Upload(weight.data.data(), weight.Size() * sizeof(float));
    CheckCudaOps(cudaMemset(sum_dev.GetData(), 0, sizeof(float)), "CudaMemset(RmsNorm sum)");    
    int n = static_cast<int>(x.Size());
    SumSquareKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(x_dev.GetData()), static_cast<float*>(sum_dev.GetData()), n);
    CheckLastKernelError("SumSquareKernel");
    RmsNormKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(x_dev.GetData()), 
                    static_cast<const float*>(w_dev.GetData()), static_cast<float*>(y_dev.GetData()), 
                    static_cast<const float*>(sum_dev.GetData()), n, eps);
    CheckLastKernelError("RmsNormKernel");

    y_dev.Download(y.data.data(), y.Size() * sizeof(float));
    return y;
}
CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight, float eps, int device_id) {
    ValidateRmsNormDeviceWeightInput(x, weight.data.data(), weight.shape, eps, device_id);
    CudaSetDevice(device_id);
    CudaDeviceBuffer w_dev(weight.Size() * sizeof(float), device_id);
    w_dev.Upload(weight.data.data(), weight.Size() * sizeof(float));
    return CudaRmsNormDeviceWeight(x, w_dev.GetData(), weight.shape, eps, device_id);
}
CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data, const std::vector<int>& weight_shape, float eps, int device_id) {
    ValidateRmsNormDeviceWeightInput(x, weight_data, weight_shape, eps, device_id);
    CudaSetDevice(device_id);

    CudaTensor y(x.Shape(), device_id);
    CudaDeviceBuffer sum_dev(sizeof(float), device_id);
    CheckCudaOps(cudaMemset(sum_dev.GetData(), 0, sizeof(float)), "CudaMemset(sum)");
    int n = static_cast<int>(x.GetSize());

    SumSquareKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(x.GetData()), static_cast<float*>(sum_dev.GetData()), n);
    CheckLastKernelError("SumSquareKernel(sum)");

    RmsNormKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(x.GetData()), static_cast<const float*>(weight_data), 
                static_cast<float*>(y.GetData()), static_cast<const float*>(sum_dev.GetData()), n, eps);
    CheckLastKernelError("RmsNormKernel(x)");

    return y;
}


Tensor CudaSilu(const Tensor& x, int device_id) {
    CudaSetDevice(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.Size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
    x_dev.Upload(x.data.data(), x.Size() * sizeof(float));

    int n = static_cast<int>(x.Size());
    SiluKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(x_dev.GetData()), static_cast<float*>(y_dev.GetData()), n);
    CheckLastKernelError("SiluKernel(x)");

    y_dev.Download(y.data.data(), y.Size() * sizeof(float));
    return y;
}
CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id) {
    if (x.GetDeviceId() != device_id) {
        throw std::runtime_error("CudaSiluDeviceInput: this tensor device is not this device");
    }
    CudaSetDevice(device_id);
    CudaTensor y(x.Shape(), device_id);
    int n = static_cast<int>(x.GetSize());

    SiluKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(x.GetData()), static_cast<float*>(y.GetData()), n);
    CheckLastKernelError("CudaSiluDeviceInput");

    return y;
}


Tensor CudaElementMutmul(const Tensor& a, const Tensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementMutmul");
    CudaSetDevice(device_id);
    Tensor y(a.shape, 0.0f);
    CudaDeviceBuffer a_dev(a.Size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.Size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
    a_dev.Upload(a.data.data(), a.Size() * sizeof(float));
    b_dev.Upload(b.data.data(), b.Size() * sizeof(float));

    int n = static_cast<int>(a.Size());
    ElementMulmutKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(a_dev.GetData()), 
                        static_cast<const float*>(b_dev.GetData()), static_cast<float*>(y_dev.GetData()), n);
    CheckLastKernelError("ElementMulmutKernel(a + b)");

    y_dev.Download(y.data.data(), y.Size() * sizeof(float));
    return y;
}
CudaTensor CudaElementMutmulDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementMutmulDeviceInput");
    if (a.GetDeviceId() != device_id || b.GetDeviceId() != device_id) {
        throw std::runtime_error("CudaElementMutmulDeviceInput: two tensor is not on this device");
    }

    CudaSetDevice(device_id);
    CudaTensor y(a.Shape(), device_id);
    int n = a.GetSize();

    ElementMulmutKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(a.GetData()), static_cast<const float*>(b.GetData()), static_cast<float*>(y.GetData()), n);
    CheckLastKernelError("ElementMulmutKernel(a * b)");

    return y;  
}


Tensor CudaElementAdd(const Tensor& a, const Tensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementAdd");
    CudaSetDevice(device_id);
    Tensor y(a.shape, 0.0f);
    CudaDeviceBuffer a_dev(a.Size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.Size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
    a_dev.Upload(a.data.data(), a.Size() * sizeof(float));
    b_dev.Upload(b.data.data(), b.Size() * sizeof(float));

    int n = static_cast<int>(a.Size());
    ElementAddKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(a_dev.GetData()), 
                        static_cast<const float*>(b_dev.GetData()), static_cast<float*>(y_dev.GetData()), n);
    CheckLastKernelError("ElementAddKernel");

    y_dev.Download(y.data.data(), y.Size() * sizeof(float));
    return y;
}
CudaTensor CudaElementAddDeviceWeight(const CudaTensor& a, const void* b_data, const std::vector<int>& b_shape, int device_id) {
    ValidateAddDeviceWeightInput(a, b_data, b_shape, device_id);
    CudaSetDevice(device_id);

    CudaTensor y(a.Shape(), device_id);
    int n = static_cast<int>(a.GetSize());

    ElementAddKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(a.GetData()), static_cast<const float*>(b_data), static_cast<float*>(y.GetData()), n);
    CheckLastKernelError("ElementAddKernel(a + b_data)");

    return y;
}
CudaTensor CudaElementAddDeviceInput(const CudaTensor& a, const CudaTensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementAddDeviceInput");
    if (a.GetDeviceId() != device_id || b.GetDeviceId() != device_id) {
        throw std::runtime_error("CudaElementAddDeviceInput: two tensor is not on this device");
    }

    CudaSetDevice(device_id);
    CudaTensor y(a.Shape(), device_id);
    int n = a.GetSize();

    ElementAddKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<const float*>(a.GetData()), static_cast<const float*>(b.GetData()), static_cast<float*>(y.GetData()), n);
    CheckLastKernelError("ElementAddKernel(a + b)");

    return y;
}


Tensor CudaSoftmax(const Tensor& x, int device_id) {
    Check1D(x, "CudaSoftmax");
    CudaSetDevice(device_id);
    Tensor y(x.shape, 0.0f);
    CudaDeviceBuffer x_dev(x.Size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
    CudaDeviceBuffer max_dev(sizeof(float), device_id);
    CudaDeviceBuffer sum_dev(sizeof(float), device_id);
    x_dev.Upload(x.data.data(), x.Size() * sizeof(float));

    int n = static_cast<int>(x.Size());
    GetMaxForSoftmaxKernel<<<1, kBlockSize>>>(static_cast<const float*>(x_dev.GetData()), static_cast<float*>(max_dev.GetData()), n);
    CheckLastKernelError("GetMaxForSoftmaxKernel(max)");

    GetExpForSoftmaxKernel<<<1, kBlockSize>>>(static_cast<const float*>(x_dev.GetData()), 
                            static_cast<float*>(y_dev.GetData()), static_cast<const float*>(max_dev.GetData()),
                            static_cast<float*>(sum_dev.GetData()), n);
    CheckLastKernelError("GetExpForSoftmaxKernel(exp)");

    SoftmaxKernel<<<GetBlockNum(n), kBlockSize>>>(static_cast<float*>(y_dev.GetData()), static_cast<const float*>(sum_dev.GetData()), n);
    CheckLastKernelError("SoftmaxKernel");

    y_dev.Download(y.data.data(), y.Size() * sizeof(float));
    return y;
}

CudaTensor CudaEmbedding(const void* embedding_data, const std::vector<int>& embedding_shape, int token_id, int device_id) {
    ValidateEmbeddingDeviceWeightInput(embedding_data, embedding_shape, token_id);
    CudaSetDevice(device_id);
    int dim = embedding_shape[1];
    CudaTensor y({dim}, device_id);
    EmbeddingLookUpKernel<<<GetBlockNum(dim), kBlockSize>>>(static_cast<const float*>(embedding_data), static_cast<float*>(y.GetData()), token_id, dim);
    CheckLastKernelError("EmbeddingLookUpKernel(token_id)");
    return y;
}

void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType type, int device_id) {
    ValidateRopeInput(q, k, pos, theta);
    CudaSetDevice(device_id);
    CudaDeviceBuffer q_dev(q.Size() * sizeof(float), device_id);
    CudaDeviceBuffer k_dev(k.Size() * sizeof(float), device_id);
    q_dev.Upload(q.data.data(), q.Size() * sizeof(float));
    k_dev.Upload(k.data.data(), k.Size() * sizeof(float));

    int n_heads = q.shape[0];
    int n_kv_heads = k.shape[0];
    int head_dim = q.shape[1];
    int n_q_pairs = n_heads * (head_dim / 2);
    int n_k_pairs = n_kv_heads * (head_dim / 2);

    if (type == RopeType::kNormal) {
        RopeNormalKernel<<<GetBlockNum(n_q_pairs), kBlockSize>>>(static_cast<float*>(q_dev.GetData()), n_heads, head_dim, pos, theta);
        CheckLastKernelError("RopeNormalKernel(q)");
        RopeNormalKernel<<<GetBlockNum(n_k_pairs), kBlockSize>>>(static_cast<float*>(k_dev.GetData()), n_kv_heads, head_dim, pos, theta);
        CheckLastKernelError("RopeNormalKernel(k)");
    } else {
        RopeNeoXKernel<<<GetBlockNum(n_q_pairs), kBlockSize>>>(static_cast<float*>(q_dev.GetData()), n_heads, head_dim, pos, theta);
        CheckLastKernelError("RopeNeoXKernel(q)");
        RopeNeoXKernel<<<GetBlockNum(n_k_pairs), kBlockSize>>>(static_cast<float*>(k_dev.GetData()), n_kv_heads, head_dim, pos, theta);
        CheckLastKernelError("RopeNeoXKernel(k)");
    }
}

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads, int head_dim, int pos, float theta, RopeType type, int device_id) {
    ValidateRopeDeviceInput(q, k, pos, theta, n_heads, n_kv_heads, head_dim, device_id);
    CudaSetDevice(device_id);

    int n_q_pairs = n_heads * (head_dim / 2);
    int n_k_pairs = n_kv_heads * (head_dim / 2);
    if (type == RopeType::kNeoX) {
        RopeNeoXKernel<<<GetBlockNum(n_q_pairs), kBlockSize>>>(static_cast<float*>(q.GetData()), n_heads, head_dim, pos, theta);
        CheckLastKernelError("RopeNeoXKernel(q)");
        RopeNeoXKernel<<<GetBlockNum(n_k_pairs), kBlockSize>>>(static_cast<float*>(k.GetData()), n_kv_heads, head_dim, pos, theta);
        CheckLastKernelError("RopeNeoXKernel(k)");
    } else {
        RopeNormalKernel<<<GetBlockNum(n_q_pairs), kBlockSize>>>(static_cast<float*>(q.GetData()), n_heads, head_dim, pos, theta);
        CheckLastKernelError("RopeNormalKernel(q)");
        RopeNormalKernel<<<GetBlockNum(n_k_pairs), kBlockSize>>>(static_cast<float*>(k.GetData()), n_kv_heads, head_dim, pos, theta);
        CheckLastKernelError("RopeNormalKernel(k)");
    }
}

}
