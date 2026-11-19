#pragma once 

#include <optional>
#include <string>
#include <vector>

#include "include/kv_cache.h"
#include "include/model.h"
#include "include/tensor.h"



namespace mini_llama {

struct BenchmarkCudaStage {
    CudaModelOpStats ops;
    CudaBufferStats buffers;
};

struct BenchmarkResult {
    int n_prompt_tokens = 0;
    int n_generated_tokens = 0;
    int n_decode_tokens = 0;
    bool used_cuda_batch_prefill = false;
    bool used_cuda_greedy = false;
    std::vector<int> generated_ids;
    double context_ms = 0.0;
    double prefill_ms = 0.0;
    std::optional<double> first_token_ms;
    double decode_ms = 0.0;
    size_t cuda_kv_bytes = 0;
    BenchmarkCudaStage cuda_context;
    BenchmarkCudaStage cuda_prefill;
    BenchmarkCudaStage cuda_first_sample;
    BenchmarkCudaStage cuda_decode;

    double total_ms() const { return first_token_ms.value_or(prefill_ms) + decode_ms; }

    double tokens_per_sec() const {
        if (total_ms() <= 0.0) {
            return 0.0;
        }
        return n_generated_tokens * 1000.0 / total_ms();
    }

    double decode_tokens_per_sec() const {
        if (decode_ms <= 0.0 || n_decode_tokens <= 0) {
            return 0.0;
        }
        return n_decode_tokens * 1000.0 / decode_ms;
    }
};


void DumpTensorShape(const Tensor& t, const std::string& name);
void DumpLogitsTopK(const Tensor& logits, int k);
void DumpKvCacheInfo(const KVCache& cache, int current_pos = -1);



BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict, unsigned int seed, bool verbose);


}
