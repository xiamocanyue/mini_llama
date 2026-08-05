#pragma once 

#include "include/tensor.h"
#include "include/cuda_data.h"


namespace mini_llama {

class CudaTensor {

public:
    CudaTensor() = default;
    explicit CudaTensor(const std::vector<int>& shape, int device_id);

    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(CudaTensor&) noexcept = delete;

    CudaTensor(CudaTensor&& other) noexcept = default;
    CudaTensor& operator=(CudaTensor&& other) noexcept = default;

    void* GetData() {
        return buffer_.GetData();
    }
    const void* GetData() const {
        return buffer_.GetData();
    }
    int GetDimNum() const {
        return static_cast<int>(shape_.size());
    }
    size_t GetSize() const {
        return element_num_;
    }
    size_t GetBytes() const {
        return buffer_.GetBytes();
    }
    int GetDeviceId() const {
        return device_id_;
    }
    bool Empty() const {
        return buffer_.Empty();
    }
    std::vector<int> Shape() const {
        return shape_;
    }

    void Reset();
    void Reset(const std::vector<int>& shape, int device_id);
    void Upload(const Tensor& src);
    Tensor Download() const;
    void DownloadTo(Tensor& dst) const;

    std::string ShapeString() const;

private:
    std::vector<int> shape_;
    size_t element_num_;
    CudaDeviceBuffer buffer_;
    int device_id_ = 0;
};



}
