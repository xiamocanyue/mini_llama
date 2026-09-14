#include "include/cuda_quant.h"


#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/cuda_data.h"

namespace mini_llama {
namespace {
constexpr int kBlockSize = 128;

int GetBlockNum(int n, int block_size) {
    return (n + kBlockSize - 1) / kBlockSize;
}
  
void CheckCudaQuant(cudaError_t err, const char* caller) {
    if (err != cudaSuccess) {
		throw std::runtime_error("CUDA quant error in " + std::string(caller) + 
								": " + cudaGetErrorString(err));
    }
}
  
void CheckLastKernel(const char* name) {
    CheckCudaQuant(cudaGetLastError(), name);
}

__device__ float HalfBitsToFloat(uint16_t bits) {
	__half_raw raw;
	raw.x = bits;
	return __half2float(raw);
}

__global__ void Q80LinearKernel(const float* x, const BlockQ80* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
	int out = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y;
	if (out >= out_features || row >= rows) {
		return;
	}

	const float* x_row = x + static_cast<size_t>(row) * in_features;
	const BlockQ80* w_row = weight + static_cast<size_t>(row) * blocks_per_row;
	float sum = 0.0f;

	for (int block_index = 0; block_index < blocks_per_row; block_index++) {
		const BlockQ80& block = w_row[block_index];
		float d = HalfBitsToFloat(block.d);
		int num_begin = block_index * kQ80BlockSize;
		int num_end = num_begin + kQ80BlockSize;
		if (num_end > in_features) {
			num_end = in_features;
		}

		for (int k = num_begin; k < num_end; k++) {
			sum += d * static_cast<float>(block.qs[k - num_begin]) * x_row[k];
		}
	}
	y[static_cast<size_t>(row) * out_features + out] = sum;
}

__device__ float Q40Value(const BlockQ40& block, int flat_index) {
	int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0F)
							: static_cast<int>(block.qs[flat_index - 16] >> 4);
	return HalfBitsToFloat(block.d) * static_cast<float>(q - 8);
}

__device__ float Q41Value(const BlockQ41& block, int flat_index) {
	int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0F)
							: static_cast<int>(block.qs[flat_index - 16] >> 4);
	return HalfBitsToFloat(block.d) * static_cast<float>(q) + HalfBitsToFloat(block.m);
}

__global__ void Q40LinearKernel(const float* x, const BlockQ40* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
	int out = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y;
	if (out >= out_features || row >= rows) {
		return;
	}

	const float* x_row = x + static_cast<size_t>(row) * in_features;
	const BlockQ40* w_row = weight + static_cast<size_t>(row) * blocks_per_row;
	float sum = 0.0f;

	for (int block_index = 0; block_index < blocks_per_row; block_index++) {
		const BlockQ40& block = w_row[block_index];
		int num_begin = block_index * kQ40BlockSize;
		int num_end = num_begin + kQ40BlockSize;
		if (num_end > in_features) {
			num_end = in_features;
		}

		for (int k = num_begin; k < num_end; k++) {
			sum += Q40Value(block, k - num_begin) * x_row[k];
		}
	}
	y[static_cast<size_t>(row) * out_features + out] = sum;
}

__global__ void Q41LinearKernel(const float* x, const BlockQ41* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
	int out = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y;
	if (out >= out_features || row >= rows) {
		return;
	}

	const float* x_row = x + static_cast<size_t>(row) * in_features;
	const BlockQ41* w_row = weight + static_cast<size_t>(row) * blocks_per_row;
	float sum = 0.0f;

	for (int block_index = 0; block_index < blocks_per_row; block_index++) {
		const BlockQ41& block = w_row[block_index];
		int num_begin = block_index * kQ41BlockSize;
		int num_end = num_begin + kQ41BlockSize;
		if (num_end > in_features) {
			num_end = in_features;
		}

		for (int k = num_begin; k < num_end; k++) {
			sum += Q41Value(block, k - num_begin) * x_row[k];
		}
	}
	y[static_cast<size_t>(row) * out_features + out] = sum;
}

void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols, const char* caller) {
	if (bias.DimNum() != 1 || bias.shape[0] != cols) {
		throw std::runtime_error(std::string(caller) + ": expected bias shape [out_features]");
	}
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			y.data[static_cast<size_t>(row) * cols + col] += bias.data[col];
		}
	}
}

int GetQuantInFeaturesFromShape(const std::vector<int>& x_shape, const std::string& x_shape_str, const char* caller) {
	if (x_shape.size() == 1) {
		return x_shape[0];
	}
	if (x_shape.size() == 2) {
		return x_shape[1];
	}
	throw std::runtime_error(
		std::string(caller) + ": expected x shape [in_features] or [batch, in_features], got " + x_shape_str);
}

void ValidateQ80LinearInputs(const Tensor& x,
                             const std::vector<int>& weight_shape,
                             size_t block_count) {
	if (weight_shape.size() != 2) {
		throw std::runtime_error("CudaQ80Linear: expected weight shape [out_features, in_features]");
	}
	int in_features = GetQuantInFeaturesFromShape(x.shape, x.ShapeString(), "CudaQ80Linear");
	if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
		throw std::runtime_error("CudaQ80Linear: empty tensors are not supported");
	}
	if (weight_shape[1] != in_features) {
		throw std::runtime_error(
			"CudaQ80Linear: dimension mismatch x=" + x.ShapeString() + " weight=[" +
			std::to_string(weight_shape[0]) + ", " +
			std::to_string(weight_shape[1]) + "]");
	}
	int blocks_per_row = GetBlockNum(in_features, kQ80BlockSize);
	size_t expected_blocks = static_cast<size_t>(weight_shape[0]) * blocks_per_row;
	if (block_count != expected_blocks) {
		throw std::runtime_error("CudaQ80Linear: block count mismatch: expected " +
								std::to_string(expected_blocks) + ", got " +
								std::to_string(block_count));
	}
}

void ValidateQuantLinearShape(const std::vector<int>& x_shape,
                              const std::string& x_shape_str,
                              const std::vector<int>& weight_shape,
                              size_t block_count, int block_size,
                              const char* caller) {
	if (weight_shape.size() != 2) {
		throw std::runtime_error(std::string(caller) + ": expected weight shape [out_features, in_features]");
	}
	int in_features = GetQuantInFeaturesFromShape(x_shape, x_shape_str, caller);
	if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
		throw std::runtime_error(std::string(caller) + ": empty tensors are not supported");
	}
	if (weight_shape[1] != in_features) {
		throw std::runtime_error(std::string(caller) +
								": dimension mismatch x=" + x_shape_str +
								" weight=[" + std::to_string(weight_shape[0]) +
								", " + std::to_string(weight_shape[1]) + "]");
	}
	int blocks_per_row = GetBlockNum(in_features, block_size);
	size_t expected_blocks = static_cast<size_t>(weight_shape[0]) * blocks_per_row;
	if (block_count != expected_blocks) {
		throw std::runtime_error(std::string(caller) +
								": block count mismatch: expected " +
								std::to_string(expected_blocks) + ", got " +
								std::to_string(block_count));
	}
}

Tensor CudaQ80LinearWithDevicePointer(const Tensor& x, const void* q8_0_weight,
                                      size_t q8_0_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {
	if (q8_0_weight == nullptr) {
		throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
	}
	ValidateQuantLinearShape(x.shape, x.ShapeString(), weight_shape, q8_0_block_count, 
							kQ80BlockSize, "CudaQ80LinearDevicePointer");

	bool input_is_1d = (x.DimNum() == 1);
	int rows = input_is_1d ? 1 : x.shape[0];
	int in_features = input_is_1d ? x.shape[0] : x.shape[1];
	int out_features = weight_shape[0];
	int blocks_per_row = GetBlockNum(in_features, kQ80BlockSize);

	Tensor y(input_is_1d ? std::vector<int>{out_features}
						: std::vector<int>{rows, out_features},
			0.0f);
	CudaDeviceBuffer x_dev(x.Size() * sizeof(float), device_id);
	CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
	x_dev.Upload(x.data.data(), x.Size() * sizeof(float));

	dim3 grid(GetBlockNum(out_features, kBlockSize), rows);
	dim3 block(kBlockSize);
	Q80LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.GetData()),
									static_cast<const BlockQ80*>(q8_0_weight),
									static_cast<float*>(y_dev.GetData()), rows,
									in_features, out_features, blocks_per_row);
	CheckLastKernel("Q80LinearKernel");

	y_dev.Download(y.data.data(), y.Size() * sizeof(float));
	if (bias != nullptr) {
		AddBiasInPlace(y, *bias, rows, out_features, "CudaQ80Linear");
	}
	return y;
}

CudaTensor CudaQ80LinearWithDeviceInput(const CudaTensor& x,
                                        const void* q8_0_weight,
                                        size_t q8_0_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
	if (q8_0_weight == nullptr) {
		throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
	}
	if (x.GetDeviceId() != device_id) {
		throw std::runtime_error("CudaQ80Linear: input tensor is on a different CUDA device");
	}
	ValidateQuantLinearShape(x.Shape(), x.ShapeString(), weight_shape, q8_0_block_count, 
							kQ80BlockSize, "CudaQ80LinearDevicePointer");

	bool input_is_1d = (x.GetDimNum() == 1);
	int rows = input_is_1d ? 1 : x.Shape()[0];
	int in_features = input_is_1d ? x.Shape()[0] : x.Shape()[1];
	int out_features = weight_shape[0];
	int blocks_per_row = GetBlockNum(in_features, kQ80BlockSize);
	CudaTensor y(input_is_1d ? std::vector<int>{out_features}
							: std::vector<int>{rows, out_features},
				device_id);

	dim3 grid(GetBlockNum(out_features, kBlockSize), rows);
	dim3 block(kBlockSize);
	Q80LinearKernel<<<grid, block>>>(static_cast<const float*>(x.GetData()),
									static_cast<const BlockQ80*>(q8_0_weight),
									static_cast<float*>(y.GetData()), rows,
									in_features, out_features, blocks_per_row);
	CheckLastKernel("Q80LinearKernel");

	return y;
}

Tensor CudaQ40LinearWithDevicePointer(const Tensor& x, const void* q4_0_weight,
                                      size_t q4_0_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {
	if (q4_0_weight == nullptr) {
		throw std::runtime_error("CudaQ40Linear : device weight pointer is null");
	}
	ValidateQuantLinearShape(x.shape, x.ShapeString(), weight_shape,
							q4_0_block_count, kQ40BlockSize, "CudaQ40Linear");

	bool input_is_1d = (x.DimNum() == 1);
	int rows = input_is_1d ? 1 : x.shape[0];
	int in_features = input_is_1d ? x.shape[0] : x.shape[1];
	int out_features = weight_shape[0];
	int blocks_per_row = GetBlockNum(in_features, kQ40BlockSize);

	Tensor y(input_is_1d ? std::vector<int>{out_features}
						: std::vector<int>{rows, out_features},
			0.0f);
	CudaDeviceBuffer x_dev(x.Size() * sizeof(float), device_id);
	CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
	x_dev.Upload(x.data.data(), x.Size() * sizeof(float));

	dim3 grid(GetBlockNum(out_features, kBlockSize), rows);
	dim3 block(kBlockSize);
	Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.GetData()),
									static_cast<const BlockQ40*>(q4_0_weight),
									static_cast<float*>(y_dev.GetData()), rows,
									in_features, out_features, blocks_per_row);
	CheckLastKernel("Q40LinearKernel");

	y_dev.Download(y.data.data(), y.Size() * sizeof(float));
	if (bias != nullptr) {
		AddBiasInPlace(y, *bias, rows, out_features, "CudaQ40Linear");
	}
	return y;
}

CudaTensor CudaQ40LinearWithDeviceInput(const CudaTensor& x,
                                        const void* q4_0_weight,
                                        size_t q4_0_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
	if (q4_0_weight == nullptr) {
		throw std::runtime_error("CudaQ40Linea: device weight pointer is null");
	}
	if (x.GetDeviceId() != device_id) {
		throw std::runtime_error("CudaQ40Linear: input tensor is on a different CUDA device");
	}
	ValidateQuantLinearShape(x.Shape(), x.ShapeString(), weight_shape,
							q4_0_block_count, kQ40BlockSize, "CudaQ40Linear");

	bool input_is_1d = (x.GetDimNum() == 1);
	int rows = input_is_1d ? 1 : x.Shape()[0];
	int in_features = input_is_1d ? x.Shape()[0] : x.Shape()[1];
	int out_features = weight_shape[0];
	int blocks_per_row = GetBlockNum(in_features, kQ40BlockSize);

	CudaTensor y(input_is_1d ? std::vector<int>{out_features}
							: std::vector<int>{rows, out_features},
				device_id);

	dim3 grid(GetBlockNum(out_features, kBlockSize), rows);
	dim3 block(kBlockSize);
	Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x.GetData()),
									static_cast<const BlockQ40*>(q4_0_weight),
									static_cast<float*>(y.GetData()), rows,
									in_features, out_features, blocks_per_row);
	CheckLastKernel("Q40LinearKernel");

	return y;
}

Tensor CudaQ41LinearWithDevicePointer(const Tensor& x, const void* q4_1_weight,
                                      size_t q4_1_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {

	if (q4_1_weight == nullptr) {
		throw std::runtime_error("CudaQ41Linear: device weight pointer is null");
	}
	ValidateQuantLinearShape(x.shape, x.ShapeString(), weight_shape,
							q4_1_block_count, kQ41BlockSize, "CudaQ41Linear");

	bool input_is_1d = (x.DimNum() == 1);
	int rows = input_is_1d ? 1 : x.shape[0];
	int in_features = input_is_1d ? x.shape[0] : x.shape[1];
	int out_features = weight_shape[0];
	int blocks_per_row = GetBlockNum(in_features, kQ41BlockSize);

	Tensor y(input_is_1d ? std::vector<int>{out_features}
						: std::vector<int>{rows, out_features},
			0.0f);
	CudaDeviceBuffer x_dev(x.Size() * sizeof(float), device_id);
	CudaDeviceBuffer y_dev(y.Size() * sizeof(float), device_id);
	x_dev.Upload(x.data.data(), x.Size() * sizeof(float));

	dim3 block(kBlockSize);
	dim3 grid(GetBlockNum(out_features, kBlockSize), rows);
	Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.GetData()),
									static_cast<const BlockQ41*>(q4_1_weight),
									static_cast<float*>(y_dev.GetData()), rows,
									in_features, out_features, blocks_per_row);
	CheckLastKernel("Q41LinearKernel");

	y_dev.Download(y.data.data(), y.Size() * sizeof(float));
	if (bias != nullptr) {
		AddBiasInPlace(y, *bias, rows, out_features, "CudaQ41Linear");
	}
	return y;
}

CudaTensor CudaQ41LinearWithDeviceInput(const CudaTensor& x,
                                        const void* q4_1_weight,
                                        size_t q4_1_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
	if (q4_1_weight == nullptr) {
		throw std::runtime_error("CudaQ41Linear: device weight pointer is null");
	}
	if (x.GetDeviceId() != device_id) {
		throw std::runtime_error("CudaQ41Linear: input tensor is on a different CUDA device");
	}
	ValidateQuantLinearShape(x.Shape(), x.ShapeString(), weight_shape,
							q4_1_block_count, kQ41BlockSize, "CudaQ41Linear");

	bool input_is_1d = x.GetDimNum() == 1;
	int rows = input_is_1d ? 1 : x.Shape()[0];
	int in_features = input_is_1d ? x.Shape()[0] : x.Shape()[1];
	int out_features = weight_shape[0];
	int blocks_per_row = GetBlockNum(in_features, kQ41BlockSize);

	CudaTensor y(input_is_1d ? std::vector<int>{out_features}
							: std::vector<int>{rows, out_features},
				device_id);

	dim3 block(kBlockSize);
	dim3 grid(GetBlockNum(out_features, kBlockSize), rows);
	Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x.GetData()),
									static_cast<const BlockQ41*>(q4_1_weight),
									static_cast<float*>(y.GetData()), rows,
									in_features, out_features, blocks_per_row);
	CheckLastKernel("Q41LinearKernel");

	return y;
}
}  // namespace

Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
						
	ValidateQ80LinearInputs(x, weight_shape, weight.size());
	CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ80), device_id);
	w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ80));
	return CudaQ80LinearWithDevicePointer(x, w_dev.GetData(), weight.size(), weight_shape, bias, device_id);
}

Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_weight, size_t q8_0_block_count, 
							const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {

	return CudaQ80LinearWithDevicePointer(x, q8_0_weight, q8_0_block_count, weight_shape, bias, device_id);
}

CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x, const void* q8_0_weight, size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape, int device_id) {

	return CudaQ80LinearWithDeviceInput(x, q8_0_weight, q8_0_block_count, weight_shape, device_id);
}

Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {

	ValidateQuantLinearShape(x.shape, x.ShapeString(), weight_shape, weight.size(), kQ40BlockSize, "CudaQ40Linear");
	CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ40), device_id);
	w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ40));
	return CudaQ40LinearWithDevicePointer(x, w_dev.GetData(), weight.size(), weight_shape, bias, device_id);
}

Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_weight, size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {
	
	return CudaQ40LinearWithDevicePointer(x, q4_0_weight, q4_0_block_count, weight_shape, bias, device_id);
}

CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x, const void* q4_0_weight, size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape, int device_id) {

	return CudaQ40LinearWithDeviceInput(x, q4_0_weight, q4_0_block_count, weight_shape, device_id);
}

Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {

	ValidateQuantLinearShape(x.shape, x.ShapeString(), weight_shape, weight.size(), kQ41BlockSize, "CudaQ41Linear");
	CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ41), device_id);
	w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ41));
	return CudaQ41LinearWithDevicePointer(x, w_dev.GetData(), weight.size(), weight_shape, bias, device_id);
}

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_weight, size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape, const Tensor* bias, int device_id) {

	return CudaQ41LinearWithDevicePointer(x, q4_1_weight, q4_1_block_count, weight_shape, bias, device_id);
}

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x, const void* q4_1_weight, size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape, int device_id) {

	return CudaQ41LinearWithDeviceInput(x, q4_1_weight, q4_1_block_count, weight_shape, device_id);
}

} 
