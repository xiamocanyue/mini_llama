#include "include/sampler.h"

#include <algorithm>

#include "include/ops.h"

namespace mini_llama {
namespace {

constexpr float kGreedyTemperature = 1e-6f;

void ValidateLogits(const Tensor& logits) {
    if (logits.data.size() == 0) {
        throw std::runtime_error("ValidateLogits: logits data size is empty");
    }
    for (auto& data : logits.data) {
        if (!std::isfinite(data)) {
            throw std::runtime_error("ValidateLogits: logits data is not finite");
        }
    }
}

void ValidateTemperature(const float temperature) {
    if (temperature < 0 || !std::isfinite(temperature)) {
        throw std::runtime_error("ValidateTemperature: temperature is must positive");
    }
}

void ValidateTopeK(const int top_k) {
    if (top_k < 0) {
        throw std::runtime_error("ValidateTopK: top_k must be positive");
    }
}

int SampleFromCandidates(const std::vector<std::pair<float, int>>& candidates, float temperature, std::mt19937& rng) {
    if (temperature < kGreedyTemperature) {
        int best = candidates[0].second;
        float best_val = candidates[0].first;
        for (auto& [val, index] : candidates) {
            if (best_val < val) {
                best_val = val;
                best = index;
            }
        }
        return best;
    }

    float max_value = candidates[0].first;
    for (auto& [value, index] : candidates) {
        if (max_value < value) {
            max_value = value;
        }
    }

    float sum = 0.0f;
    std::vector<float> probs;
    probs.reserve(candidates.size());
    for (auto& [value, index] : candidates) {
        float temp = std::exp((value - max_value) / temperature);
        sum += temp;
        probs.push_back(temp);
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float p = dist(rng);
    float cumulative = 0.0f;
    for (int i = 0; i < probs.size(); i++) {
        float temp = probs[i] / sum;
        cumulative += temp;
        if (p <= cumulative) {
            return candidates[i].second;
        }
    }
    return candidates.back().second;
}

std::vector<std::pair<float, int>> GetCandidate(const Tensor& logits) {
    std::vector<std::pair<float, int>> candidate;
    candidate.reserve(logits.data.size());
    for (size_t i = 0; i < logits.data.size(); i++) {
        candidate.emplace_back(logits.data[i], static_cast<int>(i));
    }
    return candidate;
}

} // namespace


MiniSampler::MiniSampler(unsigned int seed) {
    if (seed == 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(seed);
    }
}

MiniSampler::MiniSampler(const SamplerParams& params) : MiniSampler(params.seed) {}

int MiniSampler::SampleGreedy(const Tensor& logits) {
    ValidateLogits(logits);
    return  GetArgMax(logits);
}

int MiniSampler::SampleTemperature(const Tensor& logits, float temperature) {
    ValidateLogits(logits);
    ValidateTemperature(temperature);
    if (temperature < kGreedyTemperature) {
        return SampleGreedy(logits);
    }

    return SampleFromCandidates(GetCandidate(logits), temperature, rng_);
}

int MiniSampler::SampleTopk(const Tensor& logits, float temperature, int top_k) {
    ValidateLogits(logits);
    ValidateTopeK(top_k);
    ValidateTemperature(temperature);
    if (top_k == 0) {
        throw std::runtime_error("SampleTopk : if you use top_k, top_k must be positive");
    }
    if (top_k == 1) {
        return SampleGreedy(logits);
    }
    if (top_k > static_cast<int>(logits.data.size())) {
        top_k = static_cast<int>(logits.data.size());
    }

    std::vector<std::pair<float, int>> candidates = GetCandidate(logits);
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.first == b.first) {
            return a.second < b.second;
        }   
        return a.first > b.first;
    });

    candidates.resize(top_k);
    return SampleFromCandidates(candidates, temperature, rng_);
}

int MiniSampler::Sample(const Tensor& logits, const SamplerParams& params) {
    ValidateLogits(logits);
    ValidateTemperature(params.temperature);
    if (params.temperature < kGreedyTemperature || params.top_k == 1) {
        return SampleGreedy(logits);
    }
    if (params.top_k > 1) {
        return SampleTopk(logits, params.temperature, params.top_k);
    }
    return SampleTemperature(logits, params.temperature);
}
}