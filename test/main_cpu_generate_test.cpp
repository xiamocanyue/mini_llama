
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "include/batch.h"
#include "include/context.h"
#include "include/debug.h"
#include "include/forward.h"
#include "include/gguf_loader.h"
#include "include/gguf_tokenizer.h"
#include "include/loader.h"
#include "include/model.h"
#include "include/request_context.h"
#include "include/sampler.h"
#include "include/thread_pool.h"
#include "include/tokenizer.h"

namespace mini_llama {
namespace {

constexpr const char* kPrompt = "hello";
constexpr int kNPredict = 3;
constexpr int kLogitsTopK = 5;
constexpr unsigned int kSeed = 1;

struct CpuDemoPath {
    std::string name;
    std::string model_path;
    std::string config_path;
    std::string tokenizer_path;
    std::string merges_path;
    std::string special_path;
    bool gguf = false;
};

struct CpuDemoModel {
    MiniLlamaModel model;
    std::unique_ptr<ITokenizer> tokenizer;
};

bool ValidateLoadedDemo(const CpuDemoPath& demo, const CpuDemoModel& loaded) {
    if (!loaded.model.is_loaded) {
        std::cerr << "Failed to load " << demo.name
                << " model: " << loaded.model.load_error << "\n";
        return false;
    }
    if (!loaded.tokenizer) {
        std::cerr << "Failed to load " << demo.name << " tokenizer.\n";
        return false;
    }
    if (loaded.model.config.vocab_size < loaded.tokenizer->GetVocabSize()) {
        std::cerr << "Model vocab_size is smaller than tokenizer vocab_size: "
                << loaded.model.config.vocab_size << " < "
                << loaded.tokenizer->GetVocabSize() << "\n";
        return false;
    }
    return true;
}

void PrintTokenList(const std::string& label, const std::vector<int>& tokens) {
    std::cout << label << ": [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << tokens[i];
    }
    std::cout << "]\n";
}

void PrintModelDebug(const MiniLlamaModel& model) {
    const ModelConfig& c = model.config;
    std::cout << "config:\n"
                << "  vocab_size: " << c.vocab_size << "\n"
                << "  dim: " << c.dim << "\n"
                << "  hidden_dim: " << c.hidden_dim << "\n"
                << "  n_layers: " << c.n_layers << "\n"
                << "  n_heads: " << c.n_heads << "\n"
                << "  n_kv_heads: " << c.n_kv_heads << "\n"
                << "  head_dim: " << c.head_dim << "\n"
                << "  max_seq_len: " << c.max_seq_len << "\n"
                << "  rope_theta: " << c.rope_theta << "\n"
                << "  rms_norm_eps: " << c.rms_norm_eps << "\n";
    std::cout << "weights:\n"
                << "  actual bytes: " << ModelWeightBytes(model) << "\n"
                << "  f32 equivalent bytes: " << ModelWeightBytesF32(model)
                << "\n";
    DumpTensorShape(model.token_embedding, "token_embedding");
    DumpTensorShape(model.final_norm, "final_norm");
    if (!model.layers.empty()) {
        DumpTensorShape(model.layers[0].attention_norm, "layers.0.attention_norm");
        std::cout << "  tensor: layers.0.wq shape=" << model.layers[0].wq.ShapeString() << "\n";
        std::cout << "  tensor: layers.0.wo shape=" << model.layers[0].wo.ShapeString() << "\n";
    }
    std::cout << "  tensor: lm_head shape=" << model.lm_head.ShapeString() << "\n";
}

std::unique_ptr<ITokenizer> LoadTinyTokenizer(const std::string& path) {
    return CreateTokenizer(path);
}

std::unique_ptr<ITokenizer> LoadGgufTokenizerWithTinyFallback( const std::string& gguf_path, const std::string& fallback_vocab_path) {
    std::unique_ptr<ITokenizer> tokenizer = CreateGgufTokenizer(gguf_path);
    if (tokenizer) {
        return tokenizer;
    }
    std::cout << "GGUF tokenizer metadata unavailable; using tiny vocab fallback.\n";
    return CreateTokenizer(fallback_vocab_path);
}

CpuDemoModel LoadCpuDemoModel(const CpuDemoPath& demo) {
    CpuDemoModel loaded;
    if (demo.gguf) {
        loaded.model = LoadGgufModel(demo.model_path);
        if (loaded.model.is_loaded) {
            if (!demo.merges_path.empty()) {
                loaded.tokenizer = CreateBpeTokenizer(demo.tokenizer_path, demo.merges_path, demo.special_path);
            } else if (!demo.tokenizer_path.empty()) {
                loaded.tokenizer = LoadTinyTokenizer(demo.tokenizer_path);
            } else {
                loaded.tokenizer = LoadGgufTokenizerWithTinyFallback(demo.model_path, demo.tokenizer_path);
            }
        }
    } else {
        loaded.model = LoadModel(demo.config_path, demo.model_path);
        if (loaded.model.is_loaded) {
            loaded.tokenizer = LoadTinyTokenizer(demo.tokenizer_path);
        }
    }
    return loaded;
}


int RunCpuDemo(const CpuDemoPath& demo) {
    std::cout << "\n=== CPU demo: " << demo.name << " ===\n";
    std::cout << "model: " << demo.model_path << "\n";
    if (!demo.config_path.empty()) {
        std::cout << "config: " << demo.config_path << "\n";
    }
    std::cout << "tokenizer: " << demo.tokenizer_path << "\n";
    std::cout << "compute: cpu only\n";

    RequestContext request = StartRequest(demo.name, "cpu", demo.model_path);

    // Load model
    auto load_start = RequestClock::now();
    CpuDemoModel loaded = LoadCpuDemoModel(demo);
    request.model_load_ms = GetUseTime(load_start);
    if (!ValidateLoadedDemo(demo, loaded)) {
        request.SetError("Failed to load model or tokenizer");
        request.Finish();
        std::cout << FormatRequestTraceSummary(request) << "\n";
        return 1;
    }
    request.RecordEvent("model_load", request.model_load_ms, 0, demo.model_path);

    MiniLlamaModel& model = loaded.model;
    ITokenizer& tokenizer = *loaded.tokenizer;
    std::cout << "tokenizer:\n"
                << "  vocab_size: " << tokenizer.GetVocabSize() << "\n"
                << "  bos_id: " << tokenizer.GetBOSId() << "\n"
                << "  eos_id: " << tokenizer.GetEOSId() << "\n"
                << "  unk_id: " << tokenizer.GetUNKId() << "\n";

    // Tokenize
    auto tokenize_start = RequestClock::now();
    std::vector<int> tokens = tokenizer.Encode(kPrompt);
    request.tokenizer_ms = GetUseTime(tokenize_start);
    if (tokens.empty()) {
        request.SetError("Prompt encoded to zero tokens");
        request.Finish();
        std::cout << FormatRequestTraceSummary(request) << "\n";
        return 1;
    }
    if (tokens.size() + kNPredict > static_cast<size_t>(model.config.max_seq_len)) {
        request.SetError("Prompt plus generated tokens exceeds max_seq_len");
        request.Finish();
        std::cout << FormatRequestTraceSummary(request) << "\n";
        return 1;
    }
    request.prompt_tokens = static_cast<int>(tokens.size());
    request.RecordEvent("tokenize", request.tokenizer_ms, request.prompt_tokens, std::string("prompt=\"") + kPrompt + "\"");

    std::cout << "prompt: \"" << kPrompt << "\"\n";
    PrintTokenList("prompt_tokens", tokens);

    SetThreadCount(0);
    std::cout << "threads: " << GetThreadCount() << "\n";

    MiniLlamaContext ctx(&model);
    SamplerParams params;
    params.seed = kSeed;
    params.temperature = 0.0f;
    params.top_k = 0;
    MiniSampler sampler(params);

    // Prefill
    auto prefill_start = RequestClock::now();
    Tensor logits = ForwardBatch(ctx, model, MiniBatch::GetMultiMiniBatch(tokens, 0));
    request.prefill_ms = GetUseTime(prefill_start);
    request.prefill_tokens = static_cast<int>(tokens.size());
    ctx.n_prefill_tokens += static_cast<int>(tokens.size());
    request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens);
    DumpTensorShape(logits, "prefill_logits");

    // Decode loop
    std::vector<int> generated;
    for (int step = 0; step < kNPredict; step++) {
        auto sample_start = RequestClock::now();
        int next = sampler.Sample(logits, params);
        request.sample_ms += GetUseTime(sample_start);
        generated.push_back(next);
        tokens.push_back(next);

        std::cout << "decode_step " << step << ": token=" << next << " piece=\"" << tokenizer.DecodeToken(next) << "\"\n";
        if (next == tokenizer.GetEOSId()) {
            std::cout << "hit eos token; stopping.\n";
            break;
        }

        auto decode_start = RequestClock::now();
        logits = ForwardBatch(ctx, model, MiniBatch::GetSingleMiniBatch(next, static_cast<int>(tokens.size() - 1)));
        double decode_step_ms = GetUseTime(decode_start);
        request.decode_ms += decode_step_ms;
        ctx.n_decode_tokens++;
        request.RecordEvent("decode_step", decode_step_ms, 1, "token=" + std::to_string(next));
    }

    request.decode_tokens = static_cast<int>(generated.size());
    request.generated_tokens = static_cast<int>(generated.size());
    request.Finish();

    // Output results
    PrintTokenList("generated_tokens", generated);
    std::cout << "generated_text: \"" << tokenizer.Decode(generated) << "\"\n";
    std::cout << "\n" << FormatRequestTraceSummary(request) << "\n";
    for (const std::string& line : FormatRequestTraceEvents(request)) {
        std::cout << line << "\n";
    }
    return 0;
}


}  // namespace

} 

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " tiny|gguf\n";
        return 1;
    }
    std::string mode = argv[1];
    mini_llama::CpuDemoPath gguf_demo = {
        "tiny-gguf",
        "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf",
        "",
        "models/chat/vocab.json",
        "models/chat/merges.txt",
        "models/chat/special_tokens.json",
        true
    };
    mini_llama::CpuDemoPath tiny_demo = {
        "tiny-json-bin",
        "models/tiny/model.bin",
        "models/tiny/model.json",
        "models/tiny/vocab.json",
        "",
        "",
        false,
    };

    try {
        if (mode == "tiny") {
            return mini_llama::RunCpuDemo(tiny_demo);
        }
        if (mode == "gguf") {
            return mini_llama::RunCpuDemo(gguf_demo);
        }
    } catch (const std::exception& e) {
        std::cerr << "CPU generate test failed: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
