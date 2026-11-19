#pragma once 

#include <cstdint>
#include <string>

namespace mini_llama {

struct CudaDeviceInfo {
    int id = 0;
    std::string name;
    int compute_major;
    int compute_minor;
    size_t total_memory_bytes;
    int driver_version;
    int runtime_version;
};

bool CheckCudaRuntimeBuilt();
int GetCudaDeviceCount();
void CudaSetDevice(int device_id);
CudaDeviceInfo GetCudaDeviceInfo(int device_id);
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info);

enum class MemcpyKind {
    kHostToDevice,
    kDeviceToHost,
    kDeviceToDevice
};

void* CudaMallocBytes(size_t bytes);
void CudaFreeBytes(void* ptr);
void CudaMemcpyBytes(void* dst, const void* src, size_t bytes, MemcpyKind kind);
void CudaSynchronizeDevice(int device_id);

struct CudaBufferStats {
    size_t allocations = 0;
    size_t live_bytes = 0;
    size_t peak_bytes = 0;
    size_t h2d_copies = 0;
    size_t d2h_copies = 0;
    size_t h2d_bytes = 0;
    size_t d2h_bytes = 0;
};

// Measures buffer allocations and explicit copies on the calling thread.
class CudaBufferTracker {
public:
    explicit CudaBufferTracker(bool enabled);
    ~CudaBufferTracker();
    CudaBufferTracker(const CudaBufferTracker&) = delete;
    CudaBufferTracker& operator=(const CudaBufferTracker&) = delete;

    CudaBufferStats Snapshot() const { return stats_; }
    void ResetPeak() { stats_.peak_bytes = stats_.live_bytes; }

private:
    friend class CudaDeviceBuffer;
    friend void CudaMemcpyBytes(void*, const void*, size_t, MemcpyKind);

    static thread_local CudaBufferTracker* active_;
    CudaBufferTracker* previous_ = nullptr;
    std::uint64_t id_ = 0;
    CudaBufferStats stats_;
};


class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;
    explicit CudaDeviceBuffer(size_t bytes, int device_id = 0);
    ~CudaDeviceBuffer();

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    void Reset();
    void Reset(size_t bytes, int device_id = 0);
    void Upload(const void* src, size_t bytes);
    void Download(void* dst, size_t bytes) const;

    void* GetData() {
        return data_;
    }

    const void* GetData() const {
        return data_;
    }

    size_t GetBytes() const {
        return bytes_;
    }

    int GetDeviceId() const {
        return device_id_;
    }

    bool Empty() const {
        return data_ == nullptr;
    }


private:
    void ReleaseNoexcept() noexcept;

private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
    int device_id_ = 0;
    std::uint64_t tracking_id_ = 0;


};


}
