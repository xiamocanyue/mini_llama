#include "include/forward.h"

#include <cmath>

#include "include/ops.h"
#include "include/kv_cache.h"
#include "include/thread_pool.h"
#include "include/cuda_attention.h"
#include "include/cuda_matmul.h"
#include "include/cuda_ops.h"
#include "include/cuda_quant.h"
#include "include/cuda_tensor.h"
#include "include/cuda_kv_cache.h"

namespace mini_llama {
namespace {

bool CheckHaveGpu(const MiniLlamaModel& model) {
	if (!model.cuda_weights) {
		return false;
	}
	return true;
}

void RecordCudaLinearCall(const MiniLlamaModel& model) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->linear_calls += 1;
}

void RecordCudaHostToDeviceCopy(const MiniLlamaModel& model, size_t bytes) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->host_to_device_copies += 1;
	model.cuda_weights->host_to_device_bytes += bytes;
}

void RecordCudaDeviceToHostCopy(const MiniLlamaModel& model, size_t bytes) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->device_to_host_copies += 1;
	model.cuda_weights->device_to_host_bytes += bytes;
}

void RecordCudaLinearCopy(const MiniLlamaModel& model, const Tensor& x, const Tensor& y) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->linear_calls += 1;
	model.cuda_weights->host_to_device_copies += 1;
	model.cuda_weights->device_to_host_copies += 1;
	model.cuda_weights->host_to_device_bytes += x.Size() * sizeof(float);
	model.cuda_weights->device_to_host_bytes += y.Size() * sizeof(float);
}

void RecordCudaActivationCall(const MiniLlamaModel& model, size_t calls = 1) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->activation_calls += calls;
}

void RecordCudaAttentionCall(const MiniLlamaModel& model) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->attention_calls += 1;
}

void RecordCudaAttentionCpuFallback(const MiniLlamaModel& model) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->attention_cpu_fallbacks += 1;
}

void RecordCudaKvCacheRead(const MiniLlamaModel& model, size_t bytes) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->kv_cache_read_bytes += bytes;
}

void RecordCudaKvCacheWrite(const MiniLlamaModel& model, size_t bytes) {
	if (!CheckHaveGpu(model)) {
		return;
	}
	model.cuda_weights->kv_cache_write_bytes += bytes;
}

void ValidateForwardInputs(const MiniLlamaContext& ctx, const MiniLlamaModel& model, int token_id) {
	const ModelConfig& c = model.config;
	if (!model.is_loaded) {
		throw std::runtime_error("ForwardToken called with an unloaded model");
	}
	if (token_id < 0 || token_id >= c.vocab_size) {
		throw std::out_of_range("ForwardToken token id out of range");
	}
	if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
		throw std::out_of_range("ForwardToken position out of range");
	}
	if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
		throw std::runtime_error(
			"ForwardToken layer count does not match model config");
	}
	if (model.token_embedding.data.size() <
		static_cast<size_t>(c.vocab_size * c.dim)) {
		throw std::runtime_error(
			"ForwardToken token embedding tensor is smaller than config");
	}
}



const CudaUploadedWeight* GetCudaWeight(const MiniLlamaModel& model, const std::string& name, QuantType type) {
	if (!CheckHaveGpu(model)) {
		return nullptr;
	}
	for (const auto& weight : model.cuda_weights->weights) {
		if (weight.name == name && weight.type == type) {
			return &weight;
		}
	}
	return nullptr;
}

int GetCudaDeviceId(const MiniLlamaModel& model) {
	if (!CheckHaveGpu(model)) {
		return 0;
	}
	return model.cuda_weights->device_id;
}

bool CheckCudaLinearQuantSupport(const QuantizedTensor& weight) {
	return weight.type == QuantType::kF32 || weight.type == QuantType::kQ80 ||
		   weight.type == QuantType::kQ40 || weight.type == QuantType::kQ41;
}

bool CheckCudaLinearReady(const MiniLlamaModel& model, const std::string& name, const QuantizedTensor& weight) {
	if (CheckHaveGpu(model) && CheckCudaLinearQuantSupport(weight)
		&& GetCudaWeight(model, name, weight.type) != nullptr) {
		return true;
	}
	return false;
}

CudaTensor ForwardAddDevice(const MiniLlamaModel& model, const CudaTensor& a, const CudaTensor& b) {
	CudaTensor y_dev = CudaElementAddDeviceInput(a, b, GetCudaDeviceId(model));
	RecordCudaActivationCall(model);
	return y_dev;
}

CudaTensor UploadCudaTensor(const MiniLlamaModel& model, const Tensor& x) {
	CudaTensor x_dev = CudaTensorFromHost(x, GetCudaDeviceId(model));
	RecordCudaHostToDeviceCopy(model, x.Size() * sizeof(float));
	return x_dev;
}

Tensor DownloadCudaTensor(const MiniLlamaModel& model, const CudaTensor& x_dev) {
	Tensor y = x_dev.Download();
	RecordCudaDeviceToHostCopy(model, y.Size() * sizeof(float));
	return y;
}

CudaTensor AddOpentionBiasDevice(const MiniLlamaModel& model, const std::string& name, CudaTensor x_dev, const Tensor& bias) {
	if (bias.data.empty()) {
		return x_dev;
	}

	const CudaUploadedWeight* cuda_weight = GetCudaWeight(model, name, QuantType::kF32);
	if (cuda_weight == nullptr || cuda_weight->linear) {
		throw std::runtime_error("CudaAddBiasWeight: weight is not load");
	}
	if (cuda_weight->shape != bias.shape) {
		throw std::runtime_error("CudaAddBiasDevice: bias shape is not match");
	}

	CudaTensor y_dev = CudaElementAddDeviceWeight(x_dev, cuda_weight->buffer.GetData(), cuda_weight->shape, GetCudaDeviceId(model));
	RecordCudaActivationCall(model);
	return y_dev;
}

CudaTensor ForwardLinearDevice(const MiniLlamaModel& model, const std::string& name, 
								const CudaTensor& x, const QuantizedTensor& weight) {
    if (!CheckCudaLinearQuantSupport(weight)) {
		throw std::runtime_error("Cuda forward linear: weight quant type is not support");
	}

	const CudaUploadedWeight* cuda_weight = GetCudaWeight(model, name, weight.type);
	if (cuda_weight == nullptr) {
		throw std::runtime_error("Cuda forward Linear: weight not upload weight");
	}

	CudaTensor y;
	switch (weight.type) {
		case QuantType::kF32: 
			y = CudaLinearDeviceInput(x, cuda_weight->buffer.GetData(), cuda_weight->shape, GetCudaDeviceId(model));
			break;
		case QuantType::kQ80:
			y = CudaQ80LinearDeviceInput(x, cuda_weight->buffer.GetData(), 
					cuda_weight->block_count, cuda_weight->shape, GetCudaDeviceId(model));
			break;
		case QuantType::kQ40:
			y = CudaQ40LinearDeviceInput(x, cuda_weight->buffer.GetData(), 
					cuda_weight->block_count, cuda_weight->shape, GetCudaDeviceId(model));
			break;
		case QuantType::kQ41:
			y = CudaQ41LinearDeviceInput(x, cuda_weight->buffer.GetData(), 
					cuda_weight->block_count, cuda_weight->shape, GetCudaDeviceId(model));
			break;
	}
	RecordCudaLinearCall(model);
	return y;
}

void ForwardQkvProjectionDevice(const MiniLlamaModel& model, const std::string& layer_prefix, 
							const CudaTensor& h, const LayerWeights& lw, 
							CudaTensor& q_dev, CudaTensor& k_dev, CudaTensor& v_dev) {

    std::string q_name = layer_prefix + "wq";
    std::string k_name = layer_prefix + "wk";
    std::string v_name = layer_prefix + "wv";   
	q_dev = ForwardLinearDevice(model, q_name, h, lw.wq);
	q_dev = AddOpentionBiasDevice(model, q_name, std::move(q_dev), lw.bq);

	k_dev = ForwardLinearDevice(model, k_name, h, lw.wk);
	k_dev = AddOpentionBiasDevice(model, k_name, std::move(k_dev), lw.bk);
	
	v_dev = ForwardLinearDevice(model, v_name, h, lw.wv);
	v_dev = AddOpentionBiasDevice(model, v_name, std::move(v_dev), lw.bv);
}

CudaTensor ForwardRmsNormDevice(const MiniLlamaModel& model, const std::string& name, 
							const CudaTensor& x_dev, const Tensor& weight, float eps) {
    const CudaUploadedWeight* cuda_weight = GetCudaWeight(model, name, QuantType::kF32);
	if (cuda_weight == nullptr || cuda_weight->linear) {
		throw std::runtime_error("Cuda forward RmsNorm: weight not upload weight");
	}
	if (cuda_weight->shape != weight.shape) {
		throw std::runtime_error("Cuda Forward RmsNorm: weight shape is not match from " + name);
	}

	CudaTensor y_dev = CudaRmsNormDeviceWeight(x_dev, cuda_weight->buffer.GetData(), 
									cuda_weight->shape, eps, GetCudaDeviceId(model));
	RecordCudaAttentionCall(model);
	return y_dev;
}

void ForwardRopeDevice(const MiniLlamaModel& model, CudaTensor& q, CudaTensor& k, 
				int n_heads, int n_kv_heads, int head_dim, int pos, float theta, RopeType type) {
    CudaRopeDeviceInput(q, k, n_heads, n_kv_heads, head_dim, pos, theta, type, GetCudaDeviceId(model));
	RecordCudaActivationCall(model);
}

CudaTensor EmbedTokenDevice(const MiniLlamaModel& model, int token_id) {
	const CudaUploadedWeight* cuda_weight = GetCudaWeight(model, "token_embedding", QuantType::kF32);
	if (cuda_weight == nullptr) {
		throw std::runtime_error("Cuda embed token deivice: weight not upload weight");
	}	
	if (cuda_weight->shape != model.token_embedding.shape) {
		throw std::runtime_error("Cuda Embed Token Device: cuda_weight shape is not match");
	}
	return CudaEmbedding(cuda_weight->buffer.GetData(), cuda_weight->shape, token_id, GetCudaDeviceId(model));
}

CudaTensor ForwardAttentionDevice(const MiniLlamaModel& model, const CudaTensor& q, const CudaTensor& k, 
					const CudaTensor& v, int pos, int layer, CudaKVCache& cuda_kv_cache,
					int max_seq_len, int n_heads, int head_dim, int n_kv_heads) {
	cuda_kv_cache.WriteDevice(layer, pos, k, v);
	RecordCudaKvCacheWrite(model, (k.GetSize() + v.GetSize()) * sizeof(float));

	CudaTensor attn_result = CudaAttentionDecodeDeviceInputs(q, cuda_kv_cache, layer, pos, 
							max_seq_len, n_heads, n_kv_heads, head_dim, GetCudaDeviceId(model));
	size_t kv_cache_read_bytes = static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim) * sizeof(float) * 2;
	RecordCudaKvCacheRead(model, kv_cache_read_bytes);
	return attn_result;
}

CudaTensor ForwardFFNDevice(const MiniLlamaModel& model, const CudaTensor& h, const std::string& layer_prefix, const LayerWeights& lw) {
    std::string gate_name = layer_prefix + "w_gate";
    std::string up_name = layer_prefix + "w_up";
    std::string down_name = layer_prefix + "w_down";

	CudaTensor gate_dev = ForwardLinearDevice(model, gate_name, h, lw.w_gate);
	CudaTensor up_dev = ForwardLinearDevice(model, up_name, h, lw.w_up);
	CudaTensor silu_gate = CudaSiluDeviceInput(gate_dev, GetCudaDeviceId(model));
	CudaTensor ff_result = CudaElementMutmulDeviceInput(silu_gate, up_dev, GetCudaDeviceId(model));
	RecordCudaActivationCall(model, 2);
	return ForwardLinearDevice(model, down_name, ff_result, lw.w_down);
}

CudaTensor ForwardLayerDevice(MiniLlamaContext& ctx, const MiniLlamaModel& model, 
					const CudaTensor& x_dev, int layer, const LayerWeights& lw, const ModelConfig& config) {
	int dim = config.dim;
	int n_heads = config.n_heads;
	int n_kv_heads = config.n_kv_heads;
	int head_dim = config.head_dim;
	int pos = ctx.pos;
	int max_seq_len = config.max_seq_len;
	const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

	CudaTensor h = ForwardRmsNormDevice(model, layer_prefix + "attention_norm", x_dev, 
								lw.attention_norm, config.rms_norm_eps);
    
	CudaTensor q_dev;
	CudaTensor k_dev;
	CudaTensor v_dev;
	ForwardQkvProjectionDevice(model, layer_prefix, h, lw, q_dev, k_dev, v_dev);
	ForwardRopeDevice(model, q_dev, k_dev, n_heads, n_kv_heads, head_dim, pos, config.rope_theta, config.rope_type);
	CudaTensor attn_result = ForwardAttentionDevice(model, q_dev, k_dev, v_dev, pos, layer, 
							ctx.cuda_kv_cache_, max_seq_len, n_heads, head_dim, n_kv_heads);
	CudaTensor attn_proj = ForwardLinearDevice(model, layer_prefix + "wo", attn_result, lw.wo);
	if (attn_proj.GetSize() != static_cast<size_t>(dim)) {
		throw std::runtime_error("forward_layer_device: attention projection shape mismatch");
	}
	CudaTensor x_attn = ForwardAddDevice(model, attn_proj, x_dev);

	//FFN
	CudaTensor h2 = ForwardRmsNormDevice(model, layer_prefix + "ffn_norm", x_attn, lw.ffn_norm, config.rms_norm_eps);
	CudaTensor ff = ForwardFFNDevice(model, h2, layer_prefix, lw);
	if (ff.GetSize() != static_cast<size_t>(dim)) {
		throw std::runtime_error("forward_layer_device: FFN output shape mismatch");
	}
	CudaTensor layer_result = ForwardAddDevice(model, ff, x_attn);
	return layer_result;
}

Tensor ComputeLogitsDevice(const CudaTensor& x, const MiniLlamaModel& model, const ModelConfig& config) {
	CudaTensor norm_x = ForwardRmsNormDevice(model, "final_norm", x, model.final_norm, config.rms_norm_eps);
	CudaTensor logits_dev = ForwardLinearDevice(model, "lm_head", norm_x, model.lm_head);
	Tensor logits = DownloadCudaTensor(model, logits_dev);
	return logits.ReshapeAndCheck({config.vocab_size}, "compute_logits_device");
}



void CheckShape(const Tensor& t, const std::vector<int>& expect, const char* caller) {
    t.CheckShape(expect, caller);
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

Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a, const Tensor& b) {
	if (!CheckHaveGpu(model)) {
		if (a.shape != b.shape) {
			throw std::runtime_error("ForwardAdd: a.shape is not match b.shape");
		}

		Tensor y(a.shape, 0.0f);
		for (int i = 0; i < a.Size(); i++) {
			y.data[i] = a.data[i] + b.data[i];
		}
		return y;		
	}
	Tensor y = CudaElementAdd(a, b, GetCudaDeviceId(model));
	RecordCudaActivationCall(model);
	return y;

}

Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
    if (token_id < 0 || token_id > model.config.vocab_size) {
        throw std::runtime_error("token id is not positive");
    }
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
    if (!CheckHaveGpu(model)) {
		return Linear(x, weight);
	}
	if (!CheckCudaLinearQuantSupport(weight)) {
		return Linear(x, weight);
	}

	const CudaUploadedWeight* cuda_weight = GetCudaWeight(model, name, weight.type);
	if (cuda_weight == nullptr) {
		throw std::runtime_error("Cuda forward Linear: weight not upload weight");
	}

	Tensor y;
	switch (weight.type) {
		case QuantType::kF32: 
			y = CudaLinearDeviceWeight(x, cuda_weight->buffer.GetData(), cuda_weight->shape, nullptr, GetCudaDeviceId(model));
			break;
		case QuantType::kQ80:
			y = CudaQ80LinearDeviceWeight(x, cuda_weight->buffer.GetData(), 
					cuda_weight->block_count, cuda_weight->shape, nullptr, GetCudaDeviceId(model));
			break;
		case QuantType::kQ40:
			y = CudaQ40LinearDeviceWeight(x, cuda_weight->buffer.GetData(), 
					cuda_weight->block_count, cuda_weight->shape, nullptr, GetCudaDeviceId(model));
			break;
		case QuantType::kQ41:
			y = CudaQ41LinearDeviceWeight(x, cuda_weight->buffer.GetData(), 
					cuda_weight->block_count, cuda_weight->shape, nullptr, GetCudaDeviceId(model));
			break;
	}
	RecordCudaLinearCopy(model, x, y);
	return y;

}

void ForwardQkvProjection(const MiniLlamaModel& model, const std::string& layer_prefix, 
                            const Tensor& h, const LayerWeights& lw, 
                            Tensor& q_flat, Tensor& k_flat, Tensor& v_flat) {
    std::string q_name = layer_prefix + "wq";
    std::string k_name = layer_prefix + "wk";
    std::string v_name = layer_prefix + "wv";     
	
	if (CheckCudaLinearReady(model, q_name, lw.wq) && CheckCudaLinearReady(model, k_name, lw.wk) && CheckCudaLinearReady(model, v_name, lw.wv)) {
		CudaTensor h_dev = UploadCudaTensor(model, h);
		CudaTensor q_dev = ForwardLinearDevice(model, q_name, h_dev, lw.wq);
		CudaTensor k_dev = ForwardLinearDevice(model, k_name, h_dev, lw.wk);
		CudaTensor v_dev = ForwardLinearDevice(model, v_name, h_dev, lw.wv);

		q_flat = AddOptionalBias(DownloadCudaTensor(model, q_dev), lw.bq, "forward_layer q");
		k_flat = AddOptionalBias(DownloadCudaTensor(model, k_dev), lw.bk, "forward_layer k");
		v_flat = AddOptionalBias(DownloadCudaTensor(model, v_dev), lw.bv, "forward_layer v");

		return ;
	}
	
    q_flat = AddOptionalBias(ForwardLinear(model, h, lw.wq, q_name), lw.bq, "forward_layer q");
    k_flat = AddOptionalBias(ForwardLinear(model, h, lw.wk, k_name), lw.bk, "forward_layer k");
    v_flat = AddOptionalBias(ForwardLinear(model, h, lw.wv, v_name), lw.bv, "forward_layer v");
}

void ForwardRope(const MiniLlamaModel& model, MiniLlamaContext& ctx, Tensor& q, Tensor& k,
                 int pos, float theta, RopeType type) {
	if (!CheckHaveGpu(model)) {
		RoPE(q, k, pos, ctx.rope_cos, ctx.rope_sin, type);
	}
	CudaRope(q, k, pos, theta, type, GetCudaDeviceId(model));
	RecordCudaActivationCall(model);
}

Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
	if (!CheckHaveGpu(model)) {
    	return Softmax(x);		
	}
	Tensor y = CudaSoftmax(x, GetCudaDeviceId(model));
	return y;
}

Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate, const Tensor& up) {
    if (!CheckHaveGpu(model)) {
		return Swiglu(gate, up);
	}
	Tensor silu_gate = CudaSilu(gate, GetCudaDeviceId(model));
	Tensor y = CudaElementMutmul(silu_gate, up, GetCudaDeviceId(model));
	RecordCudaActivationCall(model, 2);
	return y;
}

int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
    return q_head / (n_heads / n_kv_heads);
}

Tensor ForwardAttention(const MiniLlamaModel& model, const Tensor& q, const Tensor& k, const Tensor& v,
                        int pos, int layer, CudaKVCache* cuda_kv_cache, int max_seq_len, 
                        MiniLlamaContext& ctx, int n_heads, int n_kv_heads, int head_dim) {

    ctx.kv_cache.Write(layer, pos, k, v);

	if (cuda_kv_cache != nullptr && !cuda_kv_cache->Empty()) {
		cuda_kv_cache->Write(layer, pos, k, v);
		RecordCudaKvCacheWrite(model, (k.Size() + v.Size()) * sizeof(float));

		Tensor attn_result = CudaAttentionDecode(q, *cuda_kv_cache, layer, pos, max_seq_len, 
										n_heads, n_kv_heads, head_dim, GetCudaDeviceId(model));
		RecordCudaAttentionCall(model);
		size_t kv_cache_read_bytes = static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim) * sizeof(float) * 2;
		RecordCudaKvCacheRead(model, kv_cache_read_bytes);
		RecordCudaHostToDeviceCopy(model, q.Size() * sizeof(float));
		RecordCudaDeviceToHostCopy(model, attn_result.Size() * sizeof(float));

		return attn_result;
	}
	if (model.cuda_weights) {
		RecordCudaAttentionCpuFallback(model);
	}

    Tensor attn_result({n_heads, head_dim}, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto compute_head = [&](int h) {
        int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

        // (q * k^T) / (sqrt(d));
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

	if (model.cuda_weights) {
		for (int h = 0; h < n_heads; h++) {
			compute_head(h);
		}
		return attn_result;
	}

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

	if (CheckCudaLinearReady(model, gate_name, lw.w_gate) && CheckCudaLinearReady(model, up_name, lw.w_up) && CheckCudaLinearReady(model, down_name, lw.w_down)) {
		CudaTensor h_dev = UploadCudaTensor(model, h);
		CudaTensor gate_dev = ForwardLinearDevice(model, gate_name, h_dev, lw.w_gate);
		CudaTensor up_dev = ForwardLinearDevice(model, up_name, h_dev, lw.w_up);
		CudaTensor silu_gate_dev = CudaSiluDeviceInput(gate_dev, GetCudaDeviceId(model));
		CudaTensor ff_dev = CudaElementMutmulDeviceInput(silu_gate_dev, up_dev, GetCudaDeviceId(model));
		RecordCudaActivationCall(model, 2);
		CudaTensor out_dev = ForwardLinearDevice(model, down_name, ff_dev, lw.w_down);
		return DownloadCudaTensor(model, out_dev);
	}
    Tensor gate = ForwardLinear(model, h, lw.w_gate, gate_name);
    Tensor up = ForwardLinear(model, h, lw.w_up, up_name);
    Tensor ffn_result = ForwardLinear(model, ForwardSwiGlu(model, gate, up), lw.w_down, down_name);
    return ffn_result;
}

Tensor ComputeLogits(const MiniLlamaModel& model, const Tensor& x, const ModelConfig& config) {
    Tensor normed = ForwardRmsNorm(model, x, model.final_norm, config.rms_norm_eps);
    Tensor logits = ForwardLinear(model, normed, model.lm_head, "lm_head");
    return logits.ReshapeAndCheck({config.vocab_size}, "computer_logits");
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
    Tensor atten_out = ForwardAttention(model, q, k, v, pos, layer, &ctx.cuda_kv_cache_, 
								model.config.max_seq_len,  ctx, n_heads, n_kv_heads, head_dim);
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


Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token_id) {
	ValidateForwardInputs(ctx, model, token_id);

    const ModelConfig& config = model.config;
    int n_layers = config.n_layers;
	if (model.cuda_weights && ctx.cuda_kv_cache_.Empty()) {
		ctx.cuda_kv_cache_.Reset(config.n_layers, config.max_seq_len, config.n_kv_heads, 
							config.head_dim, GetCudaDeviceId(model));
	}

	if (model.cuda_weights) {
		CudaTensor x_dev = EmbedTokenDevice(model, token_id);
		for (int layer = 0; layer < config.n_layers; layer++) {
			x_dev = ForwardLayerDevice(ctx, model, x_dev, layer, model.layers[layer], config);
		}
		return ComputeLogitsDevice(x_dev, model, config);
	}

    Tensor x = EmbedToken(model, token_id);
    for (int layer = 0; layer < n_layers; layer++) {
        x = ForwardLayer(ctx, model, x, layer, model.layers[layer], config);
    }
    Tensor logits = ComputeLogits(model, x, config);
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
