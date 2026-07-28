#include "include/forward.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace mini_llama;

namespace {

void Expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

QuantizedTensor MakeWeight(int rows, int cols, float scale) {
    Tensor tensor({rows, cols}, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            tensor.data[row * cols + col] = scale * static_cast<float>(1 + row + col);
        }
    }
    return ToQuantizedTensor(tensor);
}

MiniLlamaModel MakeTestModel() {
    MiniLlamaModel model;
    model.config.vocab_size = 8;
    model.config.dim = 4;
    model.config.hidden_dim = 4;
    model.config.n_layers = 1;
    model.config.n_heads = 2;
    model.config.n_kv_heads = 2;
    model.config.head_dim = 2;
    model.config.max_seq_len = 8;
    model.config.rope_theta = 10000.0f;
    model.config.rms_norm_eps = 1e-5f;

    model.token_embedding = Tensor({model.config.vocab_size, model.config.dim}, 0.0f);
    for (int token = 0; token < model.config.vocab_size; ++token) {
        for (int dim = 0; dim < model.config.dim; ++dim) {
            model.token_embedding.data[token * model.config.dim + dim] =
                0.1f * static_cast<float>(token + 1) + 0.01f * static_cast<float>(dim);
        }
    }

    model.layers.resize(model.config.n_layers);
    LayerWeights& layer = model.layers[0];
    layer.attention_norm = Tensor({model.config.dim}, 1.0f);
    layer.wq = MakeWeight(4, 4, 0.01f);
    layer.wk = MakeWeight(4, 4, 0.02f);
    layer.wv = MakeWeight(4, 4, 0.03f);
    layer.wo = MakeWeight(4, 4, 0.01f);
    layer.ffn_norm = Tensor({model.config.dim}, 1.0f);
    layer.w_gate = MakeWeight(4, 4, 0.01f);
    layer.w_up = MakeWeight(4, 4, 0.02f);
    layer.w_down = MakeWeight(4, 4, 0.01f);

    model.final_norm = Tensor({model.config.dim}, 1.0f);
    model.lm_head = MakeWeight(model.config.vocab_size, model.config.dim, 0.01f);
    model.is_loaded = true;
    return model;
}

void TestMultiTokenKvCache() {
    MiniLlamaModel model = MakeTestModel();
    MiniLlamaContext context(&model);

    Tensor first_logits = ForwardToken(context, model, 1);
    context.token_history.push_back(1);
    context.pos = 1;
    Tensor second_logits = ForwardToken(context, model, 2);
    context.token_history.push_back(2);

    const float* first_key = context.kv_cache.GetKeyPtr(0, 0, 0);
    const float* second_key = context.kv_cache.GetKeyPtr(0, 1, 0);
    bool keys_differ = false;
    for (int i = 0; i < model.config.head_dim; ++i) {
        if (std::fabs(first_key[i] - second_key[i]) > 1e-7f) {
            keys_differ = true;
            break;
        }
    }

    Expect(keys_differ, "different tokens must write different KV entries");
    Expect(context.token_history == std::vector<int>({1, 2}),
           "multi-token decode did not preserve token history");
    Expect(std::isfinite(first_logits.data[0]) && std::isfinite(second_logits.data[0]),
           "multi-token decode returned non-finite logits");
}

}  // namespace

int main() {
    try {
        TestMultiTokenKvCache();
        std::cout << "kv_cache_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kv_cache_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
