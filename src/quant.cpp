#include "include/quant.h"

#include <limits>
#include <string>
#include <cstring>
#include <cmath>

#include "include/thread_pool.h"
#include "include/ops.h"


namespace mini_llama {
namespace {

//32位浮点数 1位符号位 + 8位指数位 + 23位尾数位
//16位浮点数 1位符号位 + 5位指数位 + 10位尾数位
u_int16_t FloatToFp16(float value) {
    u_int32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    u_int32_t sign = (bits >> 16) & 0x8000u;
    u_int32_t exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    u_int32_t mantissa = bits & 0x7ffffu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<u_int16_t>(sign);
        }
        mantissa |= 0x800000u; //变成非正规数，可以从1.xx -> 0.1xx，这样可以给非常小的数保留一定的有效位 
        u_int32_t shifted = mantissa >> (1 - exponent); //右移 从1.xx -> 0.1xx ，将阶数从负数拉到0
        if ((shifted & 0x00001000u) != 0) {  // 四舍五入
            shifted += 0x00002000u;    
        }
        return static_cast<u_int16_t>(sign | (shifted >> 13));
    }
    if (exponent >= 31) {
        //指数位全为1，尾数全为0，为无穷大
        return static_cast<u_int16_t>(sign | 0x7c00u);
    }
    if ((mantissa & 0x00001000u) == 1) {
        mantissa += 0x00002000u;
        if ((mantissa & 0x00800000u) == 1) { // 产生了进位，将小数位清空，指数加一
            mantissa = 0;
            exponent++;
            if (exponent >= 31) {
                return static_cast<u_int16_t>(sign | 0x7c00u);
            }
        }
    }
    //符号位放最高位，把指数位左移10位，对齐自己的位置，之后砍掉多出来的13位尾数，填入尾数
    return static_cast<u_int16_t>(sign | (static_cast<u_int32_t>(exponent) << 10) | (mantissa >> 13));
}

float Fp16ToFloat(u_int16_t value) {
    u_int32_t sign = static_cast<u_int32_t>(value & 0x8000u) << 16;
    u_int32_t exponent = (value >> 10) & 0x1fu;
    u_int32_t mantissa = value & 0x3ffu;
    u_int32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else { //处理非正规数，找到尾数中的1， 把它移到正确的隐藏位中
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03ffu; //把开头的1重新隐去
            u_int32_t exp32 = exponent - 15 + 127;
            bits = sign | (exp32 << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        uint32_t exp32 = exponent - 15 + 127;
        bits = sign | (exp32 << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

size_t GetAndCheckNum(const std::vector<int>& shape, const char* caller) {
    size_t total = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        int dim = shape[i];
        if (dim <= 0) {
            throw std::runtime_error(std::string(caller) + "dimension at axis " + 
                                    std::to_string(i) + " must be positive, got " + std::to_string(dim));
        }
        size_t dim_size = static_cast<size_t>(dim);
        if (total > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::runtime_error(std::string(caller) + " shape element count overflow");
        }
        total *= dim_size;
    }
    return total;
}
}//namespace

//这里的静态反量化是为了测试和处理静态数据
//线性层边反量化边处理乘法，不会使用静态反量化。反量化的逻辑是一样的
std::vector<BlockQ80> QuantizeToQ80(const Tensor& t) {
    if (t.Size() == 0) {
        return {};
    }
    int row_size = (t.DimNum() >= 2) ? t.shape.back() : static_cast<int>(t.Size());
    int num_rows = (t.DimNum() >= 2) ? static_cast<int>(t.Size()) / row_size : 1;  //将不同维度的矩阵，压为2维
    int num_blocks_row = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
    size_t total_blocks = num_blocks_row * num_rows;
    std::vector<BlockQ80> blocks;
    blocks.reserve(total_blocks);

    for (int row = 0; row < num_rows; row++) {
        int row_offset = row * row_size;
        for (int row_block = 0; row_block < num_blocks_row; row_block++) {
            int block_begin = row_offset + row_block * kQ80BlockSize;
            int block_end = std::min(block_begin + kQ80BlockSize, row_offset + row_size);
            int block_len = block_end - block_begin;
            float max_abs = 0.0f;
            for (int k = block_begin; k < block_end; k++) {
                max_abs = std::max(max_abs, std::abs(t.data[k]));
            }

            BlockQ80 block;
            if (max_abs > 0.0f) {
                float d = max_abs / 127.0f;
                block.d = FloatToFp16(d);
                float store_d = Fp16ToFloat(block.d);
                float use_d = (store_d == 0.0f) ? 0.0f : 1.0f / store_d;
                for (int i = 0; i < block_len; i++) {
                    int qi = static_cast<int>(std::round(t.data[block_begin + i] * use_d));
                    if (qi > 127) {
                        qi = 127;
                    } else if (qi < -127) {
                        qi  = -127;
                    }
                    block.qs[i] = static_cast<int8_t>(qi);
                }
            } else {
                block.d = 0;
            }
            for (int i = block_len; i < kQ80BlockSize; i++) {
                block.qs[i] = 0;
            }
            blocks.push_back(block);
        }
    }
    return blocks;
}
Tensor DeQuantizeFromQ80(const std::vector<BlockQ80>& blocks, const std::vector<int>& shape) {
    size_t total = GetAndCheckNum(shape, "DeQuantizeFromQ80");
    int row_size = (shape.size() >= 2) ? shape.back() : static_cast<int>(total);
    int num_rows = (shape.size() >= 2) ? static_cast<int>(total) / row_size : 1;
    int num_blocks_row = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
    size_t total_blocks = static_cast<size_t>(num_blocks_row * num_rows); 
    if (blocks.size() != total_blocks) {
        throw std::runtime_error("DeQuantizeFormQ80: block count mismatch " + std::to_string(total_blocks) + 
                                ", got " + std::to_string(blocks.size()));
    }

    Tensor result(shape, 0.0f);
    size_t block_index = 0;
    for (int row = 0; row < num_rows; row++) {
        int row_offset = row * row_size;
        for (int row_block = 0; row_block < num_blocks_row; row_block++) {
            int block_begin = row_offset + row_block * kQ80BlockSize;
            int block_end = std::min(block_begin + kQ80BlockSize, row_offset + row_size);
            int block_len = block_end - block_begin;
            const BlockQ80& block = blocks[block_index++];
            float d = Fp16ToFloat(block.d);
            for (int k = 0; k < block_len; k++) {
                result.data[block_begin + k] = d * static_cast<float>(block.qs[k]);
            }
        }
    }
    return result;
}

std::vector<BlockQ40> QuantizeToQ40(const Tensor& t) {
    if (t.Size() == 0) {
        return {};
    }
    int row_size = (t.DimNum() >= 2) ? t.shape.back() : static_cast<int>(t.Size());
    int num_rows = (t.DimNum() >= 2) ? static_cast<int>(t.Size()) / row_size : 1;  
    int num_blocks_row = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
    size_t total_blocks = num_blocks_row * num_rows;
    std::vector<BlockQ40> blocks;
    blocks.reserve(total_blocks);

    for (int row = 0; row < num_rows; row++) {
        int row_offset = row * row_size;
        for (int row_block = 0; row_block < num_blocks_row; row_block++) {
            int block_begin = row_offset + row_block * kQ40BlockSize;
            int block_end = std::min(block_begin + kQ40BlockSize, row_offset + row_size);
            int block_len = block_end - block_begin;

            float max_abs = 0.0f;
            for (int k = block_begin; k < block_end; k++) {
                max_abs = std::max(max_abs, std::abs(t.data[k]));
            }

            BlockQ40 block;
            if (max_abs > 0.0f) {
                float d = max_abs / 7.0f;
                block.d = FloatToFp16(d);
                float store_d = Fp16ToFloat(block.d);
                float use_d = (store_d == 0.0f) ? 0.0f : 1.0f / store_d;
                for (int i = 0; i < kQ40BlockSize / 2; i++) {
                    int x0_index = block_begin + i;
                    int x1_index = block_begin + i + kQ40BlockSize / 2;
                    float x0 = (x0_index < block_end) ? (t.data[x0_index] * use_d) : 0.0f;
                    float x1 = (x1_index < block_end) ? (t.data[x1_index] * use_d) : 0.0f;
                    int qi0 = static_cast<int>(std::round(x0 + 8));
                    int qi1 = static_cast<int>(std::round(x1 + 8));
                    if (qi0 < 0) {
                        qi0 = 0;
                    }
                    if (qi0 > 15) {
                        qi0 = 15;
                    }
                    if (qi1 < 0) {
                        qi1 = 0;
                    }
                    if (qi1 > 15) {
                        qi1 = 15;
                    }
                    block.qs[i] = static_cast<u_int8_t>(qi1 << 4 | qi0);
                } 
            } else {
                    block.d = 0;
            }
            blocks.push_back(block);
        }
    }
    return blocks;
}
Tensor DeQuantizeFromQ40(const std::vector<BlockQ40>& blocks, const std::vector<int>& shape) {
    size_t total = GetAndCheckNum(shape, "DeQuantizeFromQ40");
    int row_size = (shape.size() >= 2) ? shape.back() : static_cast<int>(total);
    int num_rows = (shape.size() >= 2) ? static_cast<int>(total) / row_size : 1;
    int num_blocks_row = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
    size_t total_blocks = static_cast<size_t>(num_blocks_row * num_rows); 
    if (blocks.size() != total_blocks) {
        throw std::runtime_error("DeQuantizeFormQ40: block count mismatch " + std::to_string(total_blocks) + 
                                ", got " + std::to_string(blocks.size()));
    }

    Tensor result(shape, 0.0f);
    size_t block_index = 0;
    for (int row = 0; row < num_rows; row++) {
        int row_offset = row * row_size;
        for (int row_block = 0; row_block < num_blocks_row; row_block++) {
            int block_begin = row_offset + row_block * kQ40BlockSize;
            int block_end = std::min(block_begin + kQ40BlockSize, row_offset + row_size);
            const BlockQ40& block = blocks[block_index++];
            float d = Fp16ToFloat(block.d);
            for (int i = 0; i < kQ40BlockSize / 2; i++) {
                int q0 = static_cast<int>((block.qs[i] & 0x0fu) - 8);
                int q1 = static_cast<int>((block.qs[i] >> 4) - 8);

                int x0_index = block_begin + i;
                int x1_index = block_begin + i + kQ40BlockSize / 2;
                if (x0_index < block_end) {
                    result.data[x0_index] = d * static_cast<float>(q0);
                }
                if (x1_index < block_end) {
                    result.data[x1_index] = d * static_cast<float>(q1);
                }
            }

        }
    }
    return result;
}

Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks, const std::vector<int>& shape) {
    size_t total = GetAndCheckNum(shape, "DeQuantizeFromQ41");
    int row_size = (shape.size() >= 2) ? shape.back() : static_cast<int>(total);
    int num_rows = (shape.size() >= 2) ? static_cast<int>(total) / row_size : 1;
    int num_blocks_row = (row_size + kQ41BlockSize - 1) / kQ41BlockSize;
    size_t total_blocks = static_cast<size_t>(num_blocks_row * num_rows); 
    if (blocks.size() != total_blocks) {
        throw std::runtime_error("DeQuantizeFormQ41: block count mismatch " + std::to_string(total_blocks) + 
                                ", got " + std::to_string(blocks.size()));
    }

    Tensor result(shape, 0.0f);
    size_t block_index = 0;
    for (int row = 0; row < num_rows; row++) {
        int row_offset = row * row_size;
        for (int row_block = 0; row_block < num_blocks_row; row_block++) {
            int block_begin = row_offset + row_block * kQ40BlockSize;
            int block_end = std::min(block_begin + kQ40BlockSize, row_offset + row_size);
            const BlockQ41& block = blocks[block_index++];
            float d = Fp16ToFloat(block.d);
            float m = Fp16ToFloat(block.m);
            for (int i = 0; i < kQ40BlockSize / 2; i++) {
                int q0 = static_cast<int>((block.qs[i] & 0x0fu) - 8);
                int q1 = static_cast<int>((block.qs[i] >> 4) - 8);

                int x0_index = block_begin + i;
                int x1_index = block_begin + i + kQ41BlockSize / 2;
                if (x0_index < block_end) {
                    result.data[x0_index] = d * static_cast<float>(q0) + m;
                }
                if (x1_index < block_end) {
                    result.data[x1_index] = d * static_cast<float>(q1) + m;
                }
            }
        }
    }
    return result;
}

//x : [rows, in_feature]
//weight : [out_feature, in_feature]; 
//result : [rows, out_feature]  ->  [row, j]
template <typename BlockType, int kBlockSize>
static Tensor LinearQuantTemplate(const Tensor& x, 
                                const std::vector<BlockType>& blocks,
                                const std::vector<int>& w_shape, 
                                float (*dequant_function)(const BlockType&, int)) {

    if (w_shape.size() != 2) {
        throw std::runtime_error("LinearQuantTemplate : weight_shape expected 2D");
    }
    int in_feature;
    int rows = 1;
    bool is_1d = false;
    if (x.DimNum() == 1) {
        in_feature = x.shape[0];
        is_1d = true;
    } else if (x.DimNum() == 2) {
        in_feature = x.shape[1];
        rows = x.shape[0];
    } else {
        throw std::runtime_error("LinearQuantTemplate: X shape expected [in_feature] or [batch, in_featur]]" + x.ShapeString());
    }

    int out_feature = w_shape[0];
    if (w_shape[1] != in_feature) {
        throw std::runtime_error("LinearQuantTemplate : x shape is not match weight");
    }

    int num_blocks_row = (in_feature + kBlockSize - 1) / kBlockSize;
    size_t total_blocks = static_cast<size_t>(out_feature * num_blocks_row);
    if (blocks.size() != total_blocks) {
        throw std::runtime_error("LinearQuantTemplate: " + std::to_string(total_blocks) + ", got " + std::to_string(blocks.size()));
    }


    Tensor result(is_1d ? std::vector<int>{out_feature} : std::vector<int>{rows, out_feature}, 0.0f);

    //按照reuslt的大小来划分线程工作区间
    //result的一个格，是x的一行乘block的一行得到的
    //所以不会出现一个线程只工作到block一行的中间的任务
    ParallelFor(rows * out_feature, [&](int begin, int end) {
        for (int flat_index = begin; flat_index < end; flat_index++) {
            int row = flat_index / out_feature;
            int j = flat_index % out_feature;
            const float* x_row = x.data.data() + static_cast<size_t>(row * in_feature);
            float sum = 0.0f;
            int row_start = j * num_blocks_row;
            for (int block_index = 0; block_index < num_blocks_row; block_index++) {
                const BlockType& block = blocks[row_start + block_index];
                int block_start = block_index * kBlockSize;
                int block_end = std::min(block_start + kBlockSize, in_feature);
                for (int k = block_start; k < block_end; k++) {
                    float w = dequant_function(block, k - block_start);
                    sum += w * x_row[k];
                }
            }
            result.data[static_cast<size_t>(row * out_feature + j)] = sum;
        }
    });

    return result;
}
static float DequantQ80(const BlockQ80& block, int index) {
    float d = Fp16ToFloat(block.d);
    return d * static_cast<float>(block.qs[index]);
}
static float DequantQ40(const BlockQ40& block, int index) {
    float d = Fp16ToFloat(block.d);
    int q;
    if (index < 16) {
        q = static_cast<int>(block.qs[index] & 0x0fu) - 8;
    } else {
        q = static_cast<int>(block.qs[index - 16] >> 4) - 8;
    }
    return d * static_cast<float>(q);
}
static float DequantQ41(const BlockQ41& block, int index) {
    float d = Fp16ToFloat(block.d);
    float m = Fp16ToFloat(block.m);
    int q;
    if (index < 16) {
        q = static_cast<int>(block.qs[index] & 0x0fu);
    } else {
        q = static_cast<int>(block.qs[index - 16] >> 4);
    }
    return d * static_cast<float>(q) + m;
}

Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight, const std::vector<int>& weight_shape) {
    return LinearQuantTemplate<BlockQ80, kQ80BlockSize>(x, weight, weight_shape, DequantQ80);
}
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight, const std::vector<int>& weight_shape) {
    return LinearQuantTemplate<BlockQ40, kQ40BlockSize>(x, weight, weight_shape, DequantQ40);
}

Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight, const std::vector<int>& weight_shape) {
    return LinearQuantTemplate<BlockQ41, kQ41BlockSize>(x, weight, weight_shape, DequantQ41);
}



Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input, const std::vector<int>& weight_shape) {
    Tensor weight_f32 = DeQuantizeFromQ80(weight, weight_shape);
    return Matmul(weight_f32, input);
}
Tensor MatmulQ40(const std::vector<BlockQ40>& weight, const Tensor& input, const std::vector<int>& weight_shape) {
    Tensor weight_f32 = DeQuantizeFromQ40(weight, weight_shape);
    return Matmul(weight_f32, input);
}
float CompareMatmulError(const Tensor& w, const Tensor& input) {
    Tensor f32_result = Matmul(w, input);
    std::vector<BlockQ80> q80_w = QuantizeToQ80(w);
    Tensor q8_result = MatmulQ80(q80_w, input, w.shape);

    float max_err = 0.0f;
    for (int i = 0; i < f32_result.Size(); i++) {
        float err = std::abs(f32_result.data[i] - q8_result.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}
float CompareQ40Error(const Tensor& w, const Tensor& input) {
    Tensor f32_result = Matmul(w, input);
    std::vector<BlockQ40> q40_w = QuantizeToQ40(w);
    Tensor q40_result = MatmulQ40(q40_w, input, w.shape);

    float max_err = 0.0f;
    for (int i = 0; i < f32_result.Size(); i++) {
        float err = std::abs(f32_result.data[i] - q40_result.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

}
