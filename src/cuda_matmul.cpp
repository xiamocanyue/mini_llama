#include "include/cuda_matmul.h"

#include <string>
#include <vector>

#include "include/cuda_data.h"

#ifdef MINI_LLAMA_USE_CUDA
#include <cublas_v2.h>
#endif

namespace mini_llama {
namespace {

std::runtime_error CudaNotBuildMatmul() {
    return std::runtime_error("Cuda Matmul is not build, check with -DMINI_LLAMA_CUDA = on on a machine");
}

#ifdef MINI_LLAMA_USE_CUDA
const char* GetCublasStatusName(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_STATUS_UNKNOWN";
}

void CheckCublas(cublasStatus_t status, const char* caller) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBlase error in " + std::string(caller) + ": " + GetCublasStatusName(status));
    }
}

class CublasHandle {
public:
    explicit CublasHandle(int device_id) {
        CudaSetDevice(device_id);
        CheckCublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t Get() {
        return handle_;
    }

private:
    cublasHandle_t handle_{nullptr};

};


void ValidateMatmulShape(const Tensor& a, const Tensor& b) {
    if (a.DimNum() != 2 || b.DimNum() != 2) {
        throw std::runtime_error("CudaMatmul: tensor must be 2D");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("CudaMatmul: tensor must be match");
    }
}

void ValidateLinearinputs(const Tensor& x, const std::vector<int>& w_shape, const char* caller) {
    if (w_shape.size() != 2) {
        throw std::runtime_error(std::string(caller) + ": expect w shape is [out_feature, in_feature]");
    }

    int in_feature = 0;
    if (x.shape.size() == 1) {
        in_feature = x.shape[0];
    } else if (x.shape.size() == 2) {
        in_feature = x.shape[1];
    } else {
        throw std::runtime_error(std::string(caller) + "expect x shape [in_feature] or [batch, in_feature]");
    }
    
    if (w_shape[1] != in_feature) {
        throw std::runtime_error(std::string(caller) + "x is not match w, x in_feature is " +  
                        std::to_string(in_feature) + "w infeature is " + std::to_string(w_shape[1]));
    }
} 

void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols) {
    if (bias.DimNum() != 1 || bias.shape[0] != cols) {
        throw std::runtime_error("CudaLinear: expect bias shape [out_feature]");
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            y.data[i * cols + j] += bias.data[j];
        }
    }
}

Tensor CudaLinearWithDeviceWeight(const Tensor& x, const void* w_device, const std::vector<int>& w_shape, const Tensor* bias, int device_id) {
    if (w_device == nullptr) {
        throw std::runtime_error("CudaLinearWithDeviceWeight: device weight is null");
    }
    ValidateLinearinputs(x, w_shape, "CudaLinearWithDeviceWeight");

    bool input_1d = (x.DimNum() == 1);
    int rows = (input_1d ? 1 : x.shape[0]);
    int in_feature = (input_1d ? x.shape[0] : x.shape[1]);
    int out_feature = w_shape[0];
    
    Tensor y((input_1d ? std::vector<int>{out_feature} : std::vector<int>{rows, out_feature}), 0.0f);
    CudaDeviceBuffer x_dev(x.Size() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
    x_dev.Upload(x.data.data(), x.Size() * sizeof(float));

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    /* 
    由于cublasSgemm是以列优先，所以传入的x和w都自动发生了一次转置
    x 从[rows, in_feature] -> [in_feature, rows]
    w 从[out_feature, in_feature] -> [in_feature, out_feature]
    要的最终结果是[rows, out_feature], 所以cublasSgemm的输出结果就应该为[out_feature, rows]
    所以cublasSgemm 中的w 要再次转置 变为[out_feature, in_feature]， x不需要动即可
    */

    /*
    cublasSgemm 中的m，k，n参数来确定运算用来多少行和列
            lda，ldb，ldc来确定各自矩阵跳到下一列需要走多少步，只看物理存储，不受CUBLAS_OP_T的影响
    */
    CheckCublas(
        cublasSgemm(handle.Get(), CUBLAS_OP_T, CUBLAS_OP_N, out_feature, rows, in_feature, &alpha, 
                    static_cast<const float*>(w_device), in_feature, static_cast<const float*>(x_dev.GetData()),
                    in_feature, &beta, static_cast<float*>(y_dev.GetData()), out_feature),
        "CudaLinearWithDeviceWeight(cuBlasSgemm)"
    );

    y_dev.Download(y.data.data(), y.Size() * sizeof(float));
    if (bias != nullptr) {
        AddBiasInPlace(y, *bias, rows, out_feature);
    }
    return y;
}

CudaTensor CudaLinearDeviceInputWithDeviceWeight(const CudaTensor& x, const void* w_device, const std::vector<int>& w_shape, int device_id) {
    if (w_device == nullptr) {
        throw std::runtime_error("CudaLinearDeviceInputWithDeviceWeight: input is empty");
    }
    if (x.GetDeviceId() != device_id) {
        throw std::runtime_error("CudaLinearDeviceInputWithDeviceWeight: x device is not on same device");
    }
    ValidateLinearinputs(x.Shape(), w_shape, "CudaLinearDeviceInputWithDeviceWeight");

    bool input_1d = (x.GetDimNum() == 1);
    int rows = (input_1d ? 1 : x.Shape()[0]);
    int in_feature = (input_1d ? x.Shape()[0] : x.Shape()[1]);
    int out_feature = w_shape[0];

    CudaTensor y((input_1d ? std::vector<int>{out_feature} : std::vector<int>{rows, out_feature}), device_id);
    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CheckCublas(
        cublasSgemm(handle.Get(), CUBLAS_OP_T, CUBLAS_OP_N, out_feature, rows, in_feature,
                &alpha, static_cast<const float*>(w_device), in_feature, static_cast<const float*>(x.GetData()),
                in_feature, &beta, static_cast<float*>(y.GetData()), out_feature),
        "CudaLinearDeviceInputWithDeviceWeight(cublasSgemm)"
    );
    return y;
}

#endif
} //namespace

bool CheckCudaMatmulBuild() {
#ifdef MINI_LLAMA_USE_CUDA
  return true;
#else
  return false;
#endif
}

Tensor CudaMatmul(const Tensor& a, const Tensor& b, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    ValidateMatmulShape(a, b);

    int m = a.shape[0];
    int k = a.shape[1];
    int n = b.shape[1];
    Tensor c({m, n}, 0.0f);

    CudaDeviceBuffer a_dev(a.Size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.Size() * sizeof(float), device_id);
    CudaDeviceBuffer c_dev(c.Size() * sizeof(float), device_id);

    a_dev.Upload(a.data.data(), a.Size() * sizeof(float));
    b_dev.Upload(b.data.data(), b.Size() * sizeof(float));

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    /*
    a[m, k]; b[k, n]; c[m, n]
    cublasSgemm: a[k, m]; b[n, k]; c[n, m]; b 放前面就可以了
    */
    CheckCublas(
        cublasSgemm(handle.Get(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, static_cast<const float*>(b_dev.GetData()), n, static_cast<const float*>(a_dev.GetData()),
                k, &beta, static_cast<float*>(c_dev.GetData()), n),
        "CudaMatmul(cublasSgemm)"
    );

    c_dev.Download(c.data.data(), c.Size() * sizeof(float));
    return c;

#else
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaNotBuildMatmul();
#endif
}

Tensor CudaLinear(const Tensor& x, const Tensor& weight, const Tensor* bias, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (weight.DimNum() != 2) {
        throw std::runtime_error("CudaLinear: weight expect shape [out_feature, in_feature]");
    }

    CudaDeviceBuffer w_dev(weight.Size() * sizeof(float), device_id);
    w_dev.Upload(weight.data.data(), weight.Size() * sizeof(float));
    return CudaLinearWithDeviceWeight(x, w_dev.GetData(), weight.shape, bias, device_id);
#else
    (void)x;
    (void)weight;
    (void)bias;
    (void)device_id;
    throw CudaNotBuildMatmul();
#endif
}

Tensor CudaLinearDeviceWeight(const Tensor& x, const void* w_dev, const std::vector<int>& w_shape, const Tensor* bias, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    return CudaLinearWithDeviceWeight(x, w_dev, w_shape, bias, device_id);
#else
    (void)x;
    (void)w_dev;
    (void)w_shape;
    (void)bias;
    (void)device_id;
    throw CudaNotBuildMatmul();
#endif
}

CudaTensor CudaLinearDeviceInput(const CudaTensor& x, const void* w_dev, const std::vector<int>& w_shape, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    return CudaLinearDeviceInputWithDeviceWeight(x, w_dev, w_shape, device_id);
#else
    (void)x;
    (void)w_dev;
    (void)w_shape;
    (void)device_id;
    throw CudaNotBuildMatmul();
#endif
}

}