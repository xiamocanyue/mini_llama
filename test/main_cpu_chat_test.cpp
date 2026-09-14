
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
#include "include/prompt_builder.h"
#include "include/radix_tree.h"


namespace mini_llama {
namespace {

constexpr int kNPredict = 15;
constexpr int kLogitsTopK = 1;
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

void RunSingleMessage(PromptBuilder& builder, ChatSession& session, MiniLlamaContext& ctx, ITokenizer& tokenizer, MiniLlamaModel& model, MiniSampler& sampler) {
    // prefill
    std::string prompt = builder.Build(session.messages_);
    std::vector<int> tokens = tokenizer.Encode(prompt);
    PrintTokenList("tokens", tokens);
    session.SetTokenHistory(tokens);

    size_t cachae_len = session.GetLongestCachePrefix(tokens);
    size_t context_len = RadixTree::GetCommonPrefixLength(ctx.token_history, 0, tokens);
    size_t prefix_len = std::min(cachae_len, context_len);
    session.total_prompt_tokens_ += prefix_len;
    if (prefix_len == 0) {
        ctx = MiniLlamaContext(&model);
    } else if (prefix_len < ctx.token_history.size()) { //用户撤销对话，重新输入，要撤销cache
        ctx.token_history.resize(prefix_len);
        ctx.pos = prefix_len - 1;
    }

    std::vector<int> new_tokens(tokens.begin() + prefix_len, tokens.end());
    Tensor logits = ForwardBatch(ctx, model, MiniBatch::GetMultiMiniBatch(new_tokens, prefix_len));
    ctx.n_prefill_tokens += static_cast<int>(new_tokens.size());

    //decode
    std::vector<int> generate;    
    int generage_count = 0;
    for (int step = 0; step < kNPredict; step++) {
        int next = sampler.Sample(logits, session.sampler_params_);

        tokens.push_back(next);
        session.AppendToken(next);
        generage_count++;
        if (next != tokenizer.GetEOSId()) {
            generate.push_back(next);
        }
        logits = ForwardBatch(ctx, model, MiniBatch::GetSingleMiniBatch(next, static_cast<int>(tokens.size() - 1)));
        ctx.n_decode_tokens++;
        if (next == tokenizer.GetEOSId()) {
            break;
        }
    }


    std::string output = tokenizer.Decode(generate);
    session.AddMessage("assistant", output);
    session.Record(static_cast<int>(new_tokens.size()), generage_count, 0.0);
    session.RecordPrefix(session.token_history_);

    std::cout << session.messages_.back().role << ": " << session.messages_.back().content << '\n';
}

int RunCpuDemo(const CpuDemoPath& demo) {
    std::cout << "\n=== CPU demo: " << demo.name << " ===\n";
    std::cout << "model: " << demo.model_path << "\n";
    if (!demo.config_path.empty()) {
        std::cout << "config: " << demo.config_path << "\n";
    }
    std::cout << "tokenizer: " << demo.tokenizer_path << "\n";
    std::cout << "compute: cpu only\n";

    // Load model
    CpuDemoModel loaded = LoadCpuDemoModel(demo);
    MiniLlamaModel& model = loaded.model;
    ITokenizer& tokenizer = *loaded.tokenizer;
    std::cout << "tokenizer:\n"
                << "  vocab_size: " << tokenizer.GetVocabSize() << "\n"
                << "  bos_id: " << tokenizer.GetBOSId() << "\n"
                << "  eos_id: " << tokenizer.GetEOSId() << "\n"
                << "  unk_id: " << tokenizer.GetUNKId() << "\n";

    MiniLlamaContext ctx(&model);

    SetThreadCount(4);


    PromptBuilder builder;
    builder.SetChatTemplate("qwen2");
    ChatSession session;
    session.AddMessage("system", "you are a helper");
    session.sampler_params_.seed = kSeed;
    session.sampler_params_.temperature = 0.0f;
    session.sampler_params_.top_k = 0;

    MiniSampler sampler(session.sampler_params_);
    std::vector<int> tokens;

    std::string prompt;
    std::cout << "system: you are a helper" << "\n";
    while (true) {
        prompt.clear();
        std::cout << "开始进行下一条消息" << "user: ";
        std::cin >> prompt;
        if (prompt == "/stop") {
            std::cout << "system: stop message,";
            std::cout << "show all message from session" << '\n';
            for (auto& message : session.messages_) {
                std::cout << message.role << ": " << message.content << '\n';
            }
            std::cout << "thread count: " << GetThreadCount() << std::endl;
            break;
        }
        session.AddMessage("user", prompt);
        RunSingleMessage(builder, session, ctx, tokenizer, model, sampler);
    }
    return 0;
}

}  // namespace

} 

int main(int argc, char** argv) {
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

    if (mode == "tiny") {
        return mini_llama::RunCpuDemo(tiny_demo);
    }
    if (mode == "gguf") {
        return mini_llama::RunCpuDemo(gguf_demo);
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
