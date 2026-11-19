#pragma once 


#include "include/model.h"
#include "include/batch.h"
#include "include/context.h"
#include "include/sampler.h"

namespace mini_llama {

struct SamplingLogits {
    Tensor host;
    CudaTensor device;
};

SamplingLogits ForwardBatchForSampling(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                                       const MiniBatch& batch, const SamplerParams& params,
                                       bool require_host_logits = false);
int SampleNextToken(MiniLlamaContext& ctx, const SamplingLogits& logits,
                    MiniSampler& sampler, const SamplerParams& params);


Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token);

Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model, const MiniBatch& batch);
}