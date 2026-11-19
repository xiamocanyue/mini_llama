#include "include/cuda_attention.h"

#include <cuda_runtime_api.h>
#include <limits>

namespace mini_llama {
namespace {
constexpr int kAttentionBlockSize = 128;

void CheckCudaAttention(cudaError_t err, const char* caller) {
	if (err != cudaSuccess) {
		throw std::runtime_error("CUDA attention error in " + std::string(caller) + ": " + cudaGetErrorString(err));
	}
}

void CheckLastAttentionKernel(const char* kernel_name) {
	CheckCudaAttention(cudaGetLastError(), kernel_name);
	CheckCudaAttention(cudaDeviceSynchronize(), kernel_name);
}

size_t GetAndCheckMul(size_t a, size_t b, const char* caller) {
	if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
		throw std::runtime_error("CudaAttentionDecode: size overflow for " + std::string(caller));
	}
	return a * b;
}

void ValidateAttentionInputs(const Tensor& q, const CudaKVCache& kv_cache, int layer,
                             int pos, int n_heads, int n_kv_heads, int head_dim) {
	if (q.shape != std::vector<int>{n_heads, head_dim}) {
		throw std::runtime_error("CudaAttentionDecode: expected q shape [" + 
			std::to_string(n_heads) + ", " + std::to_string(head_dim) + "], got " + q.ShapeString());
	}
	if (kv_cache.Empty()) {
		throw std::runtime_error("CudaAttentionDecode: CUDA KV cache is empty");
	}
	if (layer < 0 || layer >= kv_cache.GetNLayers()) {
		throw std::out_of_range("CudaAttentionDecode: layer out of range");
	}
	if (pos < 0 || pos >= kv_cache.GetMaxSeqLen()) {
		throw std::out_of_range("CudaAttentionDecode: position out of range");
	}
	if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
		throw std::runtime_error("CudaAttentionDecode: head counts and head_dim must be positive");
	}
	if (n_heads % n_kv_heads != 0) {
		throw std::runtime_error("CudaAttentionDecode: n_heads must be divisible by n_kv_heads");
	}
	if (kv_cache.GetNKVHeads() != n_kv_heads || kv_cache.GetHeadDim() != head_dim) {
		throw std::runtime_error("CudaAttentionDecode: KV cache shape does not match attention shape");
	}
}


void ValidateAttentionDeviceInputs(const CudaTensor& q, const CudaKVCache& kv_cache, int layer,
                                   int pos, int n_heads, int n_kv_heads, int head_dim, int device_id) {
	if (q.GetDeviceId() != device_id) {
		throw std::runtime_error("CudaAttentionDecodeDeviceInput: q tensor is on a different CUDA device");
	}
	if (q.GetSize() != static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
		throw std::runtime_error(
			"CudaAttentionDecodeDeviceInput: q shape mismatch");
	}
	if (kv_cache.Empty()) {
		throw std::runtime_error("CudaAttentionDecodeDeviceInput: CUDA KV cache is empty");
	}
	if (layer < 0 || layer >= kv_cache.GetNLayers()) {
		throw std::out_of_range("CudaAttentionDecodeDeviceInput: layer out of range");
	}
	if (pos < 0 || pos >= kv_cache.GetMaxSeqLen()) {
		throw std::out_of_range("CudaAttentionDecodeDeviceInput: position out of range");
	}
	if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
		throw std::runtime_error("CudaAttentionDecodeDeviceInput: head counts and head_dim must be positive");
	}
	if (n_heads % n_kv_heads != 0) {
		throw std::runtime_error("CudaAttentionDecodeDeviceInput: n_heads must be divisible by n_kv_heads");
	}
	if (kv_cache.GetNKVHeads() != n_kv_heads || kv_cache.GetHeadDim() != head_dim || kv_cache.GetDeviceId() != device_id) {
		throw std::runtime_error("CudaAttentionDecodeDeviceInput: KV cache shape or device does not match attention shape");
	}
}



// (q * k^t) / sqrt(head_dim)  ->   [n_heads, pos + 1]
__global__ void AttentionScoresKernel(const float* q, const float* key, float* score, int layer,
				 					int pos, int max_seq_len, int n_heads, 
				 					int n_kv_heads, int head_dim, float scale) {
    int head = blockIdx.x;
	int index = blockIdx.y;
	if (head >= n_heads || index > pos) {
		return ;
	}

	int kv_group = n_heads / n_kv_heads;
	int kv_head_index = head / kv_group;

	size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
	// (((layer * max_seq_len) + index) * n_kv_heads) + kv_head_index) * head_dim 
	size_t key_offset = ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) + static_cast<size_t>(index))
						* static_cast<size_t>(n_kv_heads) + static_cast<size_t>(kv_head_index))
						* static_cast<size_t>(head_dim);

	__shared__ float partial[kAttentionBlockSize];
	float sum = 0.0f;
	for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
		sum += q[q_offset + i] * key[key_offset + i];
	}
	partial[threadIdx.x] = sum;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			partial[threadIdx.x] += partial[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		score[static_cast<size_t>(head) * static_cast<size_t>(pos + 1) + static_cast<size_t>(index)] = partial[0] * scale;
	}
}

//score  [n_heads, pos + 1]
// softmax() 加权求和 v -> [n_head, head_dim]
__global__ void AttentionReduceKernel(const float* score, const float* value, float* out, 
									int layer, int max_seq_len, int n_heads, 
									int n_kv_heads, int head_dim, int pos) {
	int head = blockIdx.x;
	if (head >= n_heads) {
		return ;
	}

	size_t score_base = static_cast<size_t>(head) * static_cast<size_t>(pos + 1);
	//softmax e^(x - max) / sum
	// Separate buffers prevent the sum stage from overwriting maxima
	// while other warps are still reading them.
	__shared__ float maxima[kAttentionBlockSize];
	__shared__ float sums[kAttentionBlockSize];
	float local_max = -INFINITY;
	for (int i = threadIdx.x; i <= pos; i += blockDim.x) {
		local_max = fmaxf(local_max, score[score_base + static_cast<size_t>(i)]);
	}
	maxima[threadIdx.x] = local_max;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
		}
		__syncthreads();
	}
	float max_score = maxima[0];
	float local_sum = 0.0f;
	// Empty lanes contribute zero, without evaluating -Inf - (-Inf).
	for (int i = threadIdx.x; i <= pos; i += blockDim.x) {
		local_sum += expf(score[score_base + static_cast<size_t>(i)] - max_score);
	}
	sums[threadIdx.x] = local_sum;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			sums[threadIdx.x] += sums[threadIdx.x + stride];
		}
		__syncthreads();
	}
	float sum = 1.0f / sums[0];

	size_t kv_head_index = head / (n_heads / n_kv_heads);

	for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
		float v = 0.0f;
		for (int p = 0; p <= pos; p++) {
			float scale = expf(score[score_base + static_cast<size_t>(p)] - max_score) * sum;
			//((layer * max_seq_len + p) * n_kv_heads + kv_head_index) * head_dim
			size_t value_base = ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) + static_cast<size_t>(p))
								* static_cast<size_t>(n_kv_heads) + static_cast<size_t>(kv_head_index))
								* (head_dim);
			v += scale * value[value_base + static_cast<size_t>(i)];
		}
		out[static_cast<size_t>(head) * static_cast<size_t>(head_dim) + static_cast<size_t>(i)] = v;
	}
}
constexpr int kPrefillTile = 16;
constexpr int kPrefillHeadLimit = 64;

// One block reads a KV tile for all query heads in a GQA group at one position.
__global__ void AttentionPrefillKernel(const float* query, const float* keys, const float* values,
                                      float* output, int layer, int start_pos, int cache_length,
                                      int n_heads, int n_kv_heads, int head_dim, int group_size,
                                      float scale) {
    __shared__ float key_tile[kPrefillTile * kPrefillHeadLimit];
    __shared__ float value_tile[kPrefillTile * kPrefillHeadLimit];

    const int row = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int lane = threadIdx.x % warpSize;
    const int head = kv_head * group_size + threadIdx.x / warpSize;
    const size_t q_offset = (static_cast<size_t>(row) * n_heads + head) * head_dim;
    const int upper_col = lane + warpSize;
    const float q_lo = lane < head_dim ? query[q_offset + lane] : 0.0f;
    const float q_hi = upper_col < head_dim ? query[q_offset + upper_col] : 0.0f;

    float running_max = -INFINITY;
    float normalizer = 0.0f;
    float acc_lo = 0.0f;
    float acc_hi = 0.0f;
    const int visible = start_pos + row + 1;
    const size_t layer_offset = static_cast<size_t>(layer) * cache_length;
    for (int begin = 0; begin < visible;) {
        const int count = min(kPrefillTile, visible - begin);
        for (int item = threadIdx.x; item < count * head_dim; item += blockDim.x) {
            const size_t pos = layer_offset + begin + item / head_dim;
            const size_t offset = (pos * n_kv_heads + kv_head) * head_dim + item % head_dim;
            key_tile[item] = keys[offset];
            value_tile[item] = values[offset];
        }
        __syncthreads();

        for (int i = 0; i < count; ++i) {
            const int base = i * head_dim;
            float dot = lane < head_dim ? q_lo * key_tile[base + lane] : 0.0f;
            if (upper_col < head_dim) {
                dot += q_hi * key_tile[base + upper_col];
            }
            for (int delta = warpSize / 2; delta > 0; delta /= 2) {
                dot += __shfl_down_sync(0xffffffffu, dot, delta);
            }
            const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
            const float next_max = fmaxf(running_max, score);
            const float previous_scale = expf(running_max - next_max);
            const float weight = expf(score - next_max);
            normalizer = normalizer * previous_scale + weight;
            if (lane < head_dim) {
                acc_lo = acc_lo * previous_scale + weight * value_tile[base + lane];
            }
            if (upper_col < head_dim) {
                acc_hi = acc_hi * previous_scale + weight * value_tile[base + upper_col];
            }
            running_max = next_max;
        }
        __syncthreads();
        begin += count;
    }
    if (lane < head_dim) {
        output[q_offset + lane] = acc_lo / normalizer;
    }
    if (upper_col < head_dim) {
        output[q_offset + upper_col] = acc_hi / normalizer;
    }
}
} // namespace

CudaTensor CudaAttentionPrefill(const CudaTensor& q, const CudaKVCache& kv_cache,
                               int layer, int start_pos, int n_heads) {
    const int n_kv_heads = kv_cache.GetNKVHeads();
    const int head_dim = kv_cache.GetHeadDim();
    const int cache_length = kv_cache.GetMaxSeqLen();
    const int device = kv_cache.GetDeviceId();
    const auto shape = q.Shape();
    if (!CudaPrefillAttentionSupports(n_heads, n_kv_heads, head_dim) || n_kv_heads > 65535 ||
        kv_cache.Empty() || q.Empty() || q.GetDeviceId() != device ||
        shape.size() != 2 || static_cast<size_t>(shape[1]) != static_cast<size_t>(n_heads) * head_dim ||
        layer < 0 || layer >= kv_cache.GetNLayers() || start_pos < 0 || start_pos >= cache_length ||
        shape[0] > cache_length - start_pos) {
        throw std::invalid_argument("CudaAttentionPrefill: invalid shape, device, or window");
    }

    CudaSetDevice(device);
    CudaTensor output(shape, device);
    const int group_size = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    AttentionPrefillKernel<<<dim3(shape[0], n_kv_heads), group_size * 32>>>(
        static_cast<const float*>(q.GetData()), static_cast<const float*>(kv_cache.GetKeyData()),
        static_cast<const float*>(kv_cache.GetValueData()), static_cast<float*>(output.GetData()),
        layer, start_pos, cache_length, n_heads, n_kv_heads, head_dim, group_size, scale);
    CheckCudaAttention(cudaGetLastError(), "AttentionPrefillKernel");
    return output;
}

Tensor CudaAttentionDecode(const Tensor& q, const CudaKVCache& kv_cache, int layer, 
					int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim, int device_id) {
    
	ValidateAttentionInputs(q, kv_cache, layer, pos, n_heads, n_kv_heads, head_dim);
	CudaSetDevice(device_id);
	
	size_t out_size = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
	size_t score_size = static_cast<size_t>(n_heads) * static_cast<size_t>(pos + 1);
	CudaDeviceBuffer q_dev(q.Size() * sizeof(float), device_id);
	CudaDeviceBuffer out_dev(out_size * sizeof(float), device_id);
	CudaDeviceBuffer score_dev(score_size * sizeof(float), device_id);

	q_dev.Upload(q.data.data(), q.Size() * sizeof(float));

	float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
	dim3 grid(static_cast<unsigned int>(n_heads), static_cast<unsigned int>(pos + 1));
	AttentionScoresKernel<<<grid, kAttentionBlockSize>>>(
				static_cast<const float*>(q_dev.GetData()),
				static_cast<const float*>(kv_cache.GetKeyData()),
				static_cast<float*>(score_dev.GetData()),
				layer, pos, max_seq_len, n_heads, n_kv_heads, head_dim, scale
		);
	CheckLastAttentionKernel("AttentionScoresKernel");

	AttentionReduceKernel<<<n_heads, kAttentionBlockSize>>>(
				static_cast<const float*>(score_dev.GetData()),
				static_cast<const float*>(kv_cache.GetValueData()),
				static_cast<float*>(out_dev.GetData()),
				layer, max_seq_len, n_heads, n_kv_heads, head_dim, pos
		);
	CheckLastAttentionKernel("AttentionReduceKernel");

	Tensor out({n_heads, head_dim}, 0.0f);
	out_dev.Download(out.data.data(), out_size * sizeof(float));
	return out;
}

CudaTensor CudaAttentionDecodeDeviceInputs(const CudaTensor& q, const CudaKVCache& kv_cache, int layer, 
					int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim, int device_id) {
	
	ValidateAttentionDeviceInputs(q, kv_cache, layer, pos, n_heads, n_kv_heads, head_dim, device_id);
	CudaSetDevice(device_id);

	size_t score_size = static_cast<size_t>(n_heads) * static_cast<size_t>(pos + 1); 
	size_t out_size = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
	CudaDeviceBuffer score_dev(score_size * sizeof(float), device_id);
	CudaTensor out_dev({static_cast<int>(out_size)}, device_id);

	float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
	dim3 grid(static_cast<unsigned int>(n_heads), static_cast<unsigned int>(pos + 1));
	AttentionScoresKernel<<<grid, kAttentionBlockSize>>>(
						static_cast<const float*>(q.GetData()),
						static_cast<const float*>(kv_cache.GetKeyData()),
						static_cast<float*>(score_dev.GetData()),
						layer, pos, max_seq_len, n_heads, n_kv_heads, head_dim, scale
			);
	CheckLastAttentionKernel("AttentionScoresKernel(device inputs)");

	AttentionReduceKernel<<<n_heads, kAttentionBlockSize>>>(
						static_cast<const float*>(score_dev.GetData()),
						static_cast<const float*>(kv_cache.GetValueData()),
						static_cast<float*>(out_dev.GetData()),
						layer, max_seq_len, n_heads, n_kv_heads, head_dim, pos
			);
	CheckLastAttentionKernel("AttentionReduceKernel(device inputs)");
	
	return out_dev;
}


}
