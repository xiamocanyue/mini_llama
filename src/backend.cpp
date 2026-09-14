#include "include/backend.h"

#include <stdexcept>
#include <string>

#include "include/cuda_data.h"

namespace mini_llama {

bool ParseBackendKind(const std::string& text, BackendKind& kind) {
    if (text == "cpu") {
        kind = BackendKind::kCpu;
        return true;
    }
    if (text == "cuda") {
        kind = BackendKind::kCuda;
        return true;
    }
    return false;
}

std::string BackendKindName(BackendKind kind) {
    switch (kind) {
        case BackendKind::kCpu:
            return "cpu";
        case BackendKind::kCuda:
            return "cuda";
    }
    return "unknown";
}

bool CudaBackendBuilt() { 
    return CheckCudaRuntimeBuilt(); 
}

bool CudaBackendAvailable(int device_id) {
    try {
        int count = GetCudaDeviceCount();
        return device_id >= 0 && device_id < count;
    } catch (const std::exception&) {
        return false;
    }
}

std::string CudaDeviceSummary(int device_id) {
    try {
        return CudaFormatDeviceInfo(GetCudaDeviceInfo(device_id));
    } catch (const std::exception& e) {
        return e.what();
    }
}

void ValidateBackend(const BackendConfig& config) {
    if (config.kind == BackendKind::kCpu) {
        if (config.device_id_set) {
            throw std::runtime_error("--device can only be used with --backend cuda.");
        }
        return;
    }

    if (config.kind == BackendKind::kCuda) {
        if (!CudaBackendBuilt()) {
            throw std::runtime_error(
                "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON "
                "on a NVIDIA CUDA machine.");
        }
        if (!CudaBackendAvailable(config.device_id)) {
            throw std::runtime_error(CudaDeviceSummary(config.device_id));
        }
        return;
    }

    throw std::runtime_error("unknown backend");
}

std::string BackendExecutionNote(const BackendConfig& config) {
    if (config.kind == BackendKind::kCpu) {
        return "compute: cpu";
    }
    return "compute: cuda Linear (F32/Q8_0/Q4_0/Q4_1 where uploaded) + CUDA "
            "attention over GPU KV cache + device-resident Decode path; CPU "
            "sampler";
    }

} 
