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

    __shared__ float temp[kAttentionBlockSize];
	float sum = 0.0f;
	for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
		sum += q[q_offset + i] * key[key_offset + i];
	}
	temp[threadIdx.x] = sum;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride) {
			temp[threadIdx.x] += temp[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		score[static_cast<size_t>(head) * static_cast<size_t>(pos + 1) + static_cast<size_t>(index)] = temp[0] * scale;
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
	__shared__ float max_arr[kAttentionBlockSize];
	__shared__ float sum_arr[kAttentionBlockSize];
	float max_num = -INFINITY;
	float sum_num = 0.0f;

	for (int i = threadIdx.x; i <= pos; i += blockDim.x) {
		float val = score[score_base + static_cast<size_t>(i)];
		if (val > max_num) {
			sum_num = sum_num * expf(max_num - val) + 1.0f;
			max_num = val;
		} else {
			sum_num += expf(val - max_num);
		}
	}

	max_arr[threadIdx.x] = max_num;
	sum_arr[threadIdx.x] = sum_num;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride) {
			float m1 = max_arr[threadIdx.x];
			float m2 = max_arr[threadIdx.x + stride];
			float s1 = sum_arr[threadIdx.x];
			float s2 = sum_arr[threadIdx.x + stride];

			float m_new = fmaxf(m1, m2);
			float s_new = s1 * expf(m1 - m_new) + s2 * expf(m2 - m_new);

			max_arr[threadIdx.x] = m_new;
			sum_arr[threadIdx.x] = s_new;
		}
		__syncthreads();
	}

	float max_score = max_arr[0];
	float sum = 1.0f / sum_arr[0];

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
} // namespace

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