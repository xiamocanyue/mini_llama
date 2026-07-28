#pragma once 

#include "include/kv_cache.h"
#include "include/model.h"
//#include "include/cuda_kv_cache.h"

namespace mini_llama {

struct MiniLlamaContext {
    MiniLlamaContext() = default;
    explicit MiniLlamaContext(const MiniLlamaModel* model_ptr);

    MiniLlamaContext(const MiniLlamaContext&) = delete;
    MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
    MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
    MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;

    const MiniLlamaModel* model{nullptr};
    KVCache kv_cache;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    std::vector<float> attention_scores;
    std::vector<float> attention_probs;
    //CudaKVCache cuda_kv_cache_;

    //pos用来定位当前处理的token在上下文中的绝对位置
    //定位kv_cache中的写入位置
    //定位RoPE的位置编码
    //定位attention的遍历范围
    int pos{0};

    //这里的token_history记录的是kv_cache中保存的token，是可以被重建的。
    //在forward的时候逐词添加。
    //这里还有一个关键作用，和新的prompt做GetCommonPrefixLength，在物理层面检查，是否有可复用的。
    //session的radix做不到（radix只记录“曾经算过的”，不代表当前cache中有）
    std::vector<int> token_history;

    //记录真正进入forward的prompt，即减去公共前缀的prompt
    int n_prefill_tokens{0};
    int n_decode_tokens{0};

};



}
