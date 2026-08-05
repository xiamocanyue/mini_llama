#include "include/cuda_tensor.h"

#include <string>
#include <limits>

namespace mini_llama {
namespace {

size_t CheckAndGetElementNum(const std::vector<int>& shape, const char* caller) {
    if (shape.empty()) {
        return 0;
    }

    size_t total = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        int dim = shape[i];
        if (dim <= 0) {
            throw std::runtime_error(std::string(caller) + ": shape is must positive");
        }
        size_t dim_size = static_cast<size_t>(dim);
        if (total > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::runtime_error(std::string(caller) + ": shape size is over limit");
        }
        total *= dim;
    }
    return total;
}

} // namespace

std::string CudaTensor::ShapeString() const {
    std::string out = "[";
    for (size_t i = 0; i < shape_.size(); i++) {
        if (i > 0) {
            out += ',';
        }
        out += std::to_string(shape_[i]);
    }
    out += ']';
    return out;
}

CudaTensor::CudaTensor(const std::vector<int>& shape, int device_id) : shape_(shape), device_id_(device_id) {
    Reset(shape, device_id);
} 

void CudaTensor::Reset(const std::vector<int>& shape, int device_id) {
    element_num_ = CheckAndGetElementNum(shape, "CudaTensor Reset");
    shape_ = shape;
    device_id_ = device_id;
    buffer_.Reset(element_num_ * sizeof(float), device_id_);
}

void CudaTensor::Reset() {
    shape_.clear();
    device_id_ = 0;
    element_num_ = 0;
    buffer_.Reset();
}

void CudaTensor::Upload(const Tensor& src) {
    if (src.shape != shape_) {
        throw std::runtime_error("CudaTensor Upload: shape is not match.");
    }
    buffer_.Upload(src.data.data(), src.Size() * sizeof(float));
}

void CudaTensor::DownloadTo(Tensor& dst)const {
    if (shape_ != dst.shape) {
        throw std::runtime_error("CUDA DownloadTo: CudaTensor shape is not match dst shape");
    }
    buffer_.Download(dst.data.data(), dst.Size() * sizeof(float));
}

Tensor CudaTensor::Download() const {
    if (shape_.empty()) {
        return Tensor();
    }
    Tensor dst(shape_, 0.0f);
    DownloadTo(dst);
    return dst;
}

CudaTensor GetCudaTensorFromHost(const Tensor& src, int device_id) {
    CudaTensor dst(src.shape, device_id);
    dst.Upload(src);
    return dst;
}




}