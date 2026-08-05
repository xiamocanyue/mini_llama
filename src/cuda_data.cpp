#include "include/cuda_data.h"


#include <stdexcept>
#include <sstream>
// #include <cuda_runtime_api.h>

#ifdef MINI_LLAMA_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace mini_llama {

namespace {

std::runtime_error CudaNotBuiltError() {
    return std::runtime_error("CUDA backend is not built");
}

#ifdef MINI_LLAMA_USE_CUDA
void CheckCuda(cudaError_t err, const char* caller) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA runtime error in " + std::string(caller) + ':' + cudaGetErrorString(err));
    }
}

cudaMemcpyKind ToCudaMemcpyKind(MemcpyKind kind) {
    switch (kind) {
        case MemcpyKind::kHostToDevice: 
            return cudaMemcpyHostToDevice;
        case MemcpyKind::kDeviceToHost:
            return cudaMemcpyDeviceToHost;
        case MemcpyKind::kDeviceToDevice:
            return cudaMemcpyDeviceToDevice;
    }
    throw std::runtime_error("unknow CUDA memcpy kind");
}
#endif

} //namespace

bool CheckCudaRuntimeBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else 
    return false;
#endif
}

int GetCudaDeviceCount() {
#ifdef MINI_LLAMA_USE_CUDA
    int count = 0;
    CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
#else   
    throw CudaNotBuiltError();
#endif
}

//设置工作显卡id，告知运行环境是第几个显卡
void CudaSetDevice(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    CheckCuda(cudaSetDevice(device_id), "cudaSetDevice");
#else 
    (void)device_id;
    throw CudaNotBuiltError();
#endif
}

CudaDeviceInfo GetCudaDeviceInfo(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    int count = GetCudaDeviceCount();
    if (device_id < 0 || device_id >= count) {
        throw std::runtime_error("CUDA device: " + std::to_string(device_id) + "is not available");
    }

    cudaDeviceProp prop{};
    CheckCuda(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperities");
    CudaDeviceInfo info;
    info.id = device_id;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_memory_bytes = prop.totalGlobalMem;
    CheckCuda(cudaDriverGetVersion(&info.driver_version), "cudaDeriverGetVersion");
    CheckCuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");

    return info;

#else 
    (void)device_id;
    throw CudaNotBuiltError();
#endif
}
 
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info) {
    double total_gb = static_cast<double>(info.total_memory_bytes) / (1024 * 1024 * 1024);
    std::ostringstream oss;
    oss << "device " << info.id << ": " << info.name << ", compute capability "
        << info.compute_major << '.' << info.compute_minor
        << ", total memory " << total_gb << "GB"
        << ", cuda runtime toolkit version" << info.runtime_version
        <<", cuda driver version" << info.driver_version;
    return oss.str();

}

void* CudaMallocBytes(size_t bytes) {
#ifdef MINI_LLAMA_USE_CUDA
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    CheckCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
#else 
    (void)bytes;
    throw CudaNotBuiltError();
#endif
}

void CudaFreeBytes(void* ptr) {
#ifdef MINI_LLAMA_USE_CUDA
    if (ptr == nullptr) {
        return ;
    }
    CheckCuda(cudaFree(ptr), "cudaFree");
#else 
    (void)ptr;
    throw CudaNotBuiltError();
#endif
}

void CudaMemcpyBytes(void* dst, const void* src, size_t bytes, MemcpyKind kind) {
#ifdef MINI_LLAMA_USE_CUDA
    if (bytes == 0) {
        return;
    }
    if (dst == nullptr || src == nullptr) {
        throw std::runtime_error("cuda memcpy dst or src is nullptr");
    }
    CheckCuda(cudaMemcpy(dst, src, bytes, ToCudaMemcpyKind(kind)), "cudaMemcpy");

#else 
    (void)dst;
    (void)src;
    (void)bytes;
    (void)kind;
    throw CudaNotBuiltError();
#endif
}


CudaDeviceBuffer::CudaDeviceBuffer(size_t bytes, int device_id) {
    Reset(bytes, device_id);
}

CudaDeviceBuffer::~CudaDeviceBuffer() {
    ReleaseNoexcept();
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept 
    : data_(other.data_), bytes_(other.bytes_), device_id_(other.device_id_)  {
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.device_id_ = 0;
}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    ReleaseNoexcept();
    data_ = other.data_;
    bytes_ = other.bytes_;
    device_id_ = other.device_id_;
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.device_id_ = 0;
    return *this;
}

void CudaDeviceBuffer::Reset() {
    if (data_ == nullptr) {
        return;
    }
    CudaSetDevice(device_id_);
    CudaFreeBytes(data_);
    data_ = nullptr;
    bytes_ = 0;
    device_id_ = 0;
}

void CudaDeviceBuffer::Reset(size_t bytes, int device_id) {
    Reset();
    if (bytes == 0) {
        device_id_ = device_id;
        return;
    }

    CudaSetDevice(device_id);
    CudaFreeBytes(data_);
    data_ = CudaMallocBytes(bytes);
    device_id_ = device_id;
    bytes_ = bytes;
}

void CudaDeviceBuffer::Upload(const void* src, size_t bytes) {
    if (bytes > bytes_) {
        throw std::runtime_error("CUDA upload: CUDA must more size");
    }
    CudaSetDevice(device_id_);
    CudaMemcpyBytes(data_, src, bytes, MemcpyKind::kHostToDevice);
}

void CudaDeviceBuffer::Download(void* dst, size_t bytes)const {
    if (bytes > bytes_) {
        throw std::runtime_error("CUDA download: need size is too more");
    }
    CudaSetDevice(device_id_);
    CudaMemcpyBytes(dst, data_, bytes, MemcpyKind::kDeviceToHost);
}

void CudaDeviceBuffer::ReleaseNoexcept() noexcept {
    try {
        Reset();
    } catch (...) {
        
    }
}





}
