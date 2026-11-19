#include "include/debug.h"

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <stdexcept>


#include "include/context.h"
#include "include/forward.h"
#include "include/ops.h"
#include "include/sampler.h"


namespace mini_llama {

void DumpTensorShape(const Tensor& t, const std::string& name) {
    std::cout << "  tensor: " << name;
    std::cout << " shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << t.shape[i];
    }
    std::cout << "] dtype=f32 size=" << t.Size() << "\n";
}


void DumpLogitsTopK(const Tensor& logits, int k) {
    if (logits.DimNum() != 1) {
        std::cout << "  logits: invalid shape for top-k dump\n";
        return;
    }
    int vocab_size = logits.shape[0];
    if (k <= 0 || k > vocab_size) {
        k = vocab_size;
    }

    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        pairs.push_back({logits.data[i], i});
    }

    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), [](const auto& a, const auto& b) {
         return a.first > b.first;
    });

    std::cout << "  logits top-" << k << ":\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "    [" << std::setw(3) << i << "] id=" << std::setw(4)
                << pairs[i].second << " value=" << std::fixed
                << std::setprecision(6) << pairs[i].first << "\n";
    }
}



void DumpKvCacheInfo(const KVCache& cache, int current_pos) {
    int n_layers = cache.keys.shape[0];
    int seq_len = cache.keys.shape[1];
    int n_kv_heads = cache.keys.shape[2];
    int head_dim = cache.keys.shape[3];
    std::cout << "  KV cache: layers=" << n_layers << " seq_len=" << seq_len << " n_kv_heads=" << n_kv_heads << " head_dim=" << head_dim << "\n";
    if (current_pos >= 0) {
        std::cout << "    current_pos: " << current_pos << "\n";
    }
    std::cout << "    keys shape: [" << cache.keys.shape[0] << ", "
                << cache.keys.shape[1] << ", " << cache.keys.shape[2] << ", "
                << cache.keys.shape[3] << "]\n";
    std::cout << "    values shape: [" << cache.values.shape[0] << ", "
                << cache.values.shape[1] << ", " << cache.values.shape[2] << ", "
                << cache.values.shape[3] << "]\n";
}



namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

BenchmarkCudaStage StageDifference(const BenchmarkCudaStage& end, const BenchmarkCudaStage& start) {
    auto buffers = end.buffers;
    buffers.allocations -= start.buffers.allocations;
    buffers.h2d_copies -= start.buffers.h2d_copies;
    buffers.d2h_copies -= start.buffers.d2h_copies;
    buffers.h2d_bytes -= start.buffers.h2d_bytes;
    buffers.d2h_bytes -= start.buffers.d2h_bytes;
    return {end.ops.Since(start.ops), buffers};
}

} // namespace

BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict, unsigned int seed, bool verbose) {
    const auto& config = model.config;
    if (n_predict < 0 || config.max_seq_len <= 0 || prompt_tokens.empty() ||
        prompt_tokens.size() > static_cast<size_t>(config.max_seq_len)) {
        throw std::invalid_argument("Invalid benchmark token counts");
    }

    BenchmarkResult result;
    result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());
    const int generation_limit = std::min(n_predict, config.max_seq_len - result.n_prompt_tokens);
    result.generated_ids.reserve(generation_limit);

    const bool use_cuda = ModelHasCudaWeights(model);
    CudaBufferTracker tracker(use_cuda);
    auto cuda_snapshot = [&] {
        return BenchmarkCudaStage{GetCudaModelOpStats(model), tracker.Snapshot()};
    };
    auto synchronize = [&] {
        if (use_cuda) {
            CudaSynchronizeDevice(model.cuda_weights->device_id);
        }
    };

    const auto context_start = Clock::now();
    MiniLlamaContext ctx(&model);
    if (use_cuda) {
        ctx.cuda_kv_cache_.Reset(config.n_layers, config.max_seq_len, config.n_kv_heads,
                                 config.head_dim, model.cuda_weights->device_id);
        synchronize();
    }
    result.context_ms = ElapsedMs(context_start);
    result.cuda_kv_bytes = ctx.cuda_kv_cache_.GetBytes();
    result.cuda_context.buffers = tracker.Snapshot();

    SamplerParams params;
    params.seed = seed;
    MiniSampler sampler(params);
    const MiniBatch prefill = MiniBatch::GetMultiMiniBatch(prompt_tokens, 0);
    if (verbose) {
        std::cout << "[verbose] prefill " << prompt_tokens.size() << " tokens\n";
    }

    tracker.ResetPeak();
    const auto prefill_before = cuda_snapshot();
    const auto prefill_start = Clock::now();
    SamplingLogits logits = ForwardBatchForSampling(ctx, model, prefill, params, verbose);
    result.used_cuda_batch_prefill = ctx.last_batch_used_cuda_prefill;
    result.used_cuda_greedy = generation_limit > 0 && !logits.device.Empty();
    synchronize();
    result.prefill_ms = ElapsedMs(prefill_start);
    result.cuda_prefill = StageDifference(cuda_snapshot(), prefill_before);

    if (verbose) {
        DumpTensorShape(logits.host, "logits");
        DumpLogitsTopK(logits.host, 5);
        DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
    }

    if (generation_limit > 0) {
        tracker.ResetPeak();
        const auto sample_before = cuda_snapshot();
        const auto sample_start = Clock::now();
        int next_token = SampleNextToken(ctx, logits, sampler, params);
        result.first_token_ms = result.prefill_ms + ElapsedMs(sample_start);
        result.cuda_first_sample = StageDifference(cuda_snapshot(), sample_before);
        result.generated_ids.push_back(next_token);

        if (verbose) {
            std::cout << "[verbose] First token=" << next_token << "\n";
        }

        tracker.ResetPeak();
        const auto decode_before = cuda_snapshot();
        for (int i = 1; i < generation_limit; ++i) {
            const auto decode_start = Clock::now();
            const MiniBatch decode = MiniBatch::GetSingleMiniBatch(next_token, result.n_prompt_tokens + i - 1);
            logits = ForwardBatchForSampling(ctx, model, decode, params, verbose);
            next_token = SampleNextToken(ctx, logits, sampler, params);
            if (i + 1 == generation_limit) {
                synchronize();
            }
            result.decode_ms += ElapsedMs(decode_start);
            result.generated_ids.push_back(next_token);
            ++result.n_decode_tokens;

            if (verbose) {
                std::cout << "[verbose] Decode step " << i << ": token=" << next_token << "\n";
                DumpLogitsTopK(logits.host, 5);
            }
        }
        if (result.n_decode_tokens > 0) {
            result.cuda_decode = StageDifference(cuda_snapshot(), decode_before);
        }
    }
    result.n_generated_tokens = static_cast<int>(result.generated_ids.size());

    if (verbose) {
        DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
    }

    return result;
}


}
