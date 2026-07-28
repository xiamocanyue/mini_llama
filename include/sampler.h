#pragma once 

#include <random>

#include "include/tensor.h"

namespace mini_llama {

struct SamplerParams {
    float temperature = 0.0f; // 0 为贪心选取, 用来调整采样的平滑度，
                              //数值越大，低分的词输出的概率越大，输出越随机
    int top_k = 0;           // 0 表示禁用top_k
    unsigned int seed = 0;   // 0 表示用系统的seed
};

class MiniSampler {
public:
    explicit MiniSampler(unsigned int seed = 0);
    explicit MiniSampler(const SamplerParams& params);

    int Sample(const Tensor& logits, const SamplerParams& params);
    static int SampleGreedy(const Tensor& logits);
    int SampleTemperature(const Tensor& logits, float temperature);
    int SampleTopk(const Tensor& logits, float temperature, int top_k);


private:
    std::mt19937 rng_;
};

}