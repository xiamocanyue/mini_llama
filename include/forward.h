#pragma once 


#include "include/model.h"
#include "include/batch.h"
#include "include/context.h"

namespace mini_llama {

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token);

Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model, const MiniBatch& batch);
}