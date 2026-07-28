#include "include/forward.h"

#include <cmath>

#include "include/ops.h"
#include "include/kv_cache.h"
#include "include/thread_pool.h"

namespace mini_llama {
namespace {

void CheckShape(const Tensor& t, const std::vector<int>& expect, const char* caller) {
    t.CheckShape(expect, caller);
}

Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("ForwardAdd: a.shape is not match b.shape");
    }

    Tensor y(a.shape, 0.0f);
    for (int i = 0; i < a.Size(); i++) {
        y.data[i] = a.data[i] + b.data[i];
    }
    return y;
}

Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
    int dim = model.config.dim;
    Tensor x({dim}, 0.0f);
    for (int i = 0; i < dim; i++) {
        x.data[i] = model.token_embedding.data[token_id * dim + i];
    }
    return x;
}

Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x, const Tensor& weight, float eps) {
    return RmsNorm(x, weight, eps);
}

Tensor ForwardLinear(const MiniLlamaModel& model, const Tensor& x, const QuantizedTensor& weight, const std::string& name) {
    return Linear(x, weight);
}

Tensor AddOptionalBias(const Tensor& x, const Tensor& bias, const char* caller) {
    if (bias.data.empty()) {
        return x;
    }
    if (x.DimNum() != 1 || bias.DimNum() != 1 || x.shape[0] != bias.shape[0]) {
        throw std::runtime_error(std::string(caller) + "bias shape is not match x shape");
    }

    Tensor y = x;
    for (int i = 0; i < x.shape[0]; i++) {
        y.data[i] += bias.data[i];
    }
    return y;
}

void ForwardQkvProjection(const MiniLlamaModel& model, const std::string& layer_prefix, 
                            const Tensor& h, const LayerWeights& lw, 
                            Tensor& q_flat, Tensor& k_flat, Tensor& v_flat) {
    std::string q_name = layer_prefix + "wq";
    std::string k_name = layer_prefix + "wk";
    std::string v_name = layer_prefix + "wv";                            
    q_flat = AddOptionalBias(ForwardLinear(model, h, lw.wq, q_name), lw.bq, "forward_layer q");
    k_flat = AddOptionalBias(ForwardLinear(model, h, lw.wk, k_name), lw.bk, "forward_layer k");
    v_flat = AddOptionalBias(ForwardLinear(model, h, lw.wv, v_name), lw.bv, "forward_layer v");
}

void ForwardRope(const MiniLlamaModel& model, MiniLlamaContext& ctx, Tensor& q, Tensor& k,
                 int pos, float theta, RopeType type) {
    RoPE(q, k, pos, ctx.rope_cos, ctx.rope_sin, type);
}

Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
    return Softmax(x);
}

Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate, const Tensor& up) {
    return Swiglu(gate, up);
}


int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
    return q_head / (n_heads / n_kv_heads);
}

Tensor ForwardAttention(const MiniLlamaModel& model, const Tensor& q, const Tensor& k, const Tensor& v,
                        int pos, int layer, // CudaKvCache* cuda_kv_cache 
                        MiniLlamaContext& ctx, int n_heads, int n_kv_heads, int head_dim) {

    ctx.kv_cache.Write(layer, pos, k, v);

    Tensor attn_result({n_heads, head_dim}, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto compute_head = [&](int h) {
        int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

        // softmax(q * k^T) / (sqrt(d));
        float* scores_data = ctx.attention_scores.data() + h * model.config.max_seq_len;
        float* probs_data = ctx.attention_probs.data() + h * model.config.max_seq_len;
        for (int p = 0; p <= pos; p++) {  //只遍历看到了的token，天然的实现了mask（）
            const float* k_ptr = ctx.kv_cache.GetKeyPtr(layer, p, kv_head);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q.data[h * head_dim + d] * k_ptr[d];
            }
            scores_data[p] = dot * scale;
        }
        //softmax
        float max_score = scores_data[0];
        for (int p = 1; p <= pos; ++p) {
            max_score = std::max(max_score, scores_data[p]);
        }
        float prob_sum = 0.0f;
        for (int p = 0; p <= pos; ++p) {
            probs_data[p] = std::exp(scores_data[p] - max_score);
            prob_sum += probs_data[p];
        }
        for (int p = 0; p <= pos; ++p) {
            probs_data[p] /= prob_sum;
        }

        //根据得到的比例，合并v
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int p = 0; p <= pos; p++) {
                const float* v_ptr = ctx.kv_cache.GetValuePtr(layer, p, kv_head);
                val += probs_data[p] * v_ptr[d];  // k * 矩阵一列，再相加
            }
            attn_result.data[h * head_dim + d] = val;
        }
    };

    ParallelFor(n_heads, [&](int begin, int end) {
        for (int h = begin; h < end; h++) {
            compute_head(h);
        }
    });

    return attn_result;
}

Tensor ForwardFFN(const MiniLlamaModel& model, const std::string& layer_prefix, const Tensor& h, const LayerWeights& lw) {
    std::string gate_name = layer_prefix + "w_gate";
    std::string up_name = layer_prefix + "w_up";
    std::string down_name = layer_prefix + "w_down";

    Tensor gate = ForwardLinear(model, h, lw.w_gate, gate_name);
    Tensor up = ForwardLinear(model, h, lw.w_up, up_name);
    Tensor ffn_result = ForwardLinear(model, ForwardSwiGlu(model, gate, up), lw.w_down, down_name);
    return ffn_result;
}

Tensor ComputerLogits(const MiniLlamaModel& model, const Tensor& x, const ModelConfig& config) {
    Tensor normed = ForwardRmsNorm(model, x, model.final_norm, config.rms_norm_eps);
    //
    Tensor logits_flat = ForwardLinear(model, normed, model.lm_head, "lm_head");
    return logits_flat.ReshapeAndCheck({config.vocab_size}, "computer_logits");
}

/*
一层神经网络： RmsNorm（） -> 得到qkv -> RoPE 添加位置信息 -> 自注意力计算和缓存更新 -> 输出拼接 -> 残差链接 -> RmsNorm（）-> FNN前馈神经网络 -> logits
    FNN前馈神经网络： Gate 和 Up -> 激活函数 -> 获得结果 -> 残差连接

*/
Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model, 
                const Tensor& x, int layer, const LayerWeights& lw, const ModelConfig& config) {

    int dim = config.dim;
    int n_heads = config.n_heads;
    int n_kv_heads = config.n_kv_heads;
    int head_dim = config.head_dim;
    int pos = ctx.pos;
    const std::string layer_prefix = "layer" + std::to_string(layer) + ".";
    //运算前先进行RmsNorm计算
    Tensor h = ForwardRmsNorm(model, x, lw.attention_norm, config.rms_norm_eps);
    //获得qkv
    Tensor q_flat, k_flat, v_flat;
    ForwardQkvProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);
    Tensor q = q_flat.ReshapeAndCheck({n_heads, head_dim}, "forward_layer q");
    Tensor k = k_flat.ReshapeAndCheck({n_kv_heads, head_dim}, "forward_layer k");
    Tensor v = v_flat.ReshapeAndCheck({n_kv_heads, head_dim}, "forward_layer v");
    //给qk加入位置信息
    ForwardRope(model, ctx, q, k, pos, config.rope_theta, config.rope_type);
    //执行自注意力计算
    //atten_out [n_heads, head_dim]
    Tensor atten_out = ForwardAttention(model, q, k, v, pos, layer, ctx, n_heads, n_kv_heads, head_dim);
    //输出拼接
    Tensor atten_out_flat = atten_out.ReshapeAndCheck({1, n_heads * head_dim}, "forward_layer atten_out");
    Tensor atten_proj = ForwardLinear(model, atten_out_flat, lw.wo, layer_prefix + "wo");

    CheckShape(atten_proj, {1, dim}, "forward_layer attn_proj");
    Tensor atten_proj_1d = atten_proj.ReshapeAndCheck({dim}, "forward_layer atten_proj 1d");
    
    //残差连接
    Tensor x_atten = ForwardAdd(model, x, atten_proj_1d);

    //FNN 前馈神经网络
    Tensor h2 = ForwardRmsNorm(model, x_atten, lw.ffn_norm, config.rms_norm_eps);
    Tensor ff = ForwardFFN(model, layer_prefix, h2, lw);
    CheckShape(ff, {dim}, "forward_layer ffn");
    
    //残差连接
    return ForwardAdd(model, x_atten, ff);
}


}//namespace 


Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
    const ModelConfig& config = model.config;
    int n_layers = config.n_layers;

    Tensor x = EmbedToken(model, token);
    for (int layer = 0; layer < n_layers; layer++) {
        x = ForwardLayer(ctx, model, x, layer, model.layers[layer], config);
    }
    Tensor logits = ComputerLogits(model, x, config);
    return logits;
}

//处理一组token，逐个把token记录在上下文中，但只用返回最后一个token的logits，用来预测下一个token
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model, const MiniBatch& batch) {
    if (batch.GetTokensNum() == 0) {
        throw std::runtime_error("ForwardBatch:batch is empty");
    }
    if (batch.tokens.size() != batch.positions.size()) {
        throw std::runtime_error("ForwardBatch: batch.token size is not match position size");
    }

    Tensor logits;
    for (int i = 0; i < batch.GetTokensNum(); i++) {
        ctx.pos = batch.positions[i];
        logits = ForwardToken(ctx, model, batch.tokens[i]);
        ctx.token_history.push_back(batch.tokens[i]);
    }
    return logits;
}

}
