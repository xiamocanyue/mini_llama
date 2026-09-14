#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "include/backend.h"
#include "include/batch.h"
#include "include/chat.h"
#include "include/kv_cache.h"
#include "include/loader.h"
#include "include/matmul_dispatch.h"
#include "include/ops.h"
#include "include/prompt_builder.h"
#include "include/quant.h"
#include "include/radix_tree.h"
#include "include/request_context.h"
#include "include/sampler.h"
#include "include/tensor.h"
#include "include/thread_pool.h"
#include "include/tokenizer.h"

using namespace mini_llama;

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void ExpectNear(float actual, float expected, float tolerance,
                const std::string& message) {
  if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

template <typename Fn>
void ExpectThrows(Fn&& fn, const std::string& message) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

void TestTensor() {
  Tensor tensor({2, 3}, 0.0f);
  tensor.At2(1, 2) = 7.5f;
  Expect(tensor.Size() == 6, "Tensor size mismatch");
  ExpectNear(tensor.At2(1, 2), 7.5f, 0.0f, "Tensor index mismatch");
  Tensor reshaped = tensor.ReshapeAndCheck({3, 2}, "TestTensor");
  Expect(reshaped.shape == std::vector<int>({3, 2}),
         "Tensor reshape mismatch");
  ExpectNear(reshaped.data[5], 7.5f, 0.0f,
             "Tensor reshape changed data");
  ExpectThrows([] { Tensor invalid({2, -1}, 0.0f); },
               "negative Tensor dimension was accepted");
  ExpectThrows([&] { (void)tensor.At2(2, 0); },
               "Tensor out-of-range index was accepted");
}

void TestMatmulAndLinear() {
  Tensor a({2, 3}, 0.0f);
  a.data = {1, 2, 3, 4, 5, 6};
  Tensor b({3, 2}, 0.0f);
  b.data = {7, 8, 9, 10, 11, 12};
  Tensor product = Matmul(a, b);
  Expect(product.shape == std::vector<int>({2, 2}),
         "Matmul shape mismatch");
  ExpectNear(product.At2(0, 0), 58.0f, 1e-5f, "Matmul value mismatch");
  ExpectNear(product.At2(1, 1), 154.0f, 1e-5f,
             "Matmul value mismatch");

  Tensor x({3}, 0.0f);
  x.data = {2, 3, 4};
  Tensor weight({2, 3}, 0.0f);
  weight.data = {1, 0, 0, 0, 2, 0};
  Tensor linear = Linear(x, weight);
  Expect(linear.shape == std::vector<int>({2}), "Linear shape mismatch");
  ExpectNear(linear.data[0], 2.0f, 1e-5f, "Linear value mismatch");
  ExpectNear(linear.data[1], 6.0f, 1e-5f, "Linear value mismatch");
  ExpectThrows([&] { (void)Matmul(a, a); },
               "Matmul dimension mismatch was accepted");
}

void TestElementwiseOneDimensional() {
  Tensor a({4}, 0.0f);
  a.data = {-1, 0, 1, 2};
  Tensor b({4}, 0.0f);
  b.data = {2, 3, 4, 5};
  Tensor multiplied = ElementMutmul(a, b);
  Expect(multiplied.data == std::vector<float>({-2, 0, 4, 10}),
         "ElementMutmul 1D mismatch");
  Tensor silu = Silu(a);
  ExpectNear(silu.data[1], 0.0f, 1e-6f, "SiLU zero mismatch");
  Tensor swiglu = Swiglu(a, b);
  ExpectNear(swiglu.data[2], silu.data[2] * 4.0f, 1e-6f,
             "SwiGLU mismatch");
}

void TestElementwiseMultiDimensional() {
  Tensor a({2, 2}, 2.0f);
  Tensor b({2, 2}, 3.0f);
  Tensor multiplied = ElementMutmul(a, b);
  Tensor silu = Silu(a);
  Tensor swiglu = Swiglu(a, b);
  for (size_t i = 0; i < a.Size(); ++i) {
    ExpectNear(multiplied.data[i], 6.0f, 1e-6f,
               "ElementMutmul did not process every 2D element");
    ExpectNear(silu.data[i], 2.0f / (1.0f + std::exp(-2.0f)), 1e-6f,
               "SiLU did not process every 2D element");
    ExpectNear(swiglu.data[i], silu.data[i] * 3.0f, 1e-6f,
               "SwiGLU did not process every 2D element");
  }
}

void TestNormalizationAndRope() {
  Tensor x({4}, 1.0f);
  Tensor weight({4}, 2.0f);
  Tensor norm = RmsNorm(x, weight, 1e-5f);
  for (float value : norm.data) {
    ExpectNear(value, 2.0f, 1e-4f, "RMSNorm mismatch");
  }

  Tensor logits({3}, 0.0f);
  logits.data = {1, 2, 3};
  Tensor probabilities = Softmax(logits);
  float sum = std::accumulate(probabilities.data.begin(),
                              probabilities.data.end(), 0.0f);
  ExpectNear(sum, 1.0f, 1e-6f, "Softmax sum mismatch");
  Expect(GetArgMax(probabilities) == 2, "ArgMax mismatch");

  Tensor q({1, 4}, 0.0f);
  q.data = {1, 2, 3, 4};
  Tensor k = q;
  std::vector<float> cos_table = {1, 1};
  std::vector<float> sin_table = {0, 0};
  RoPE(q, k, 0, cos_table, sin_table, RopeType::kNormal);
  Expect(q.data == std::vector<float>({1, 2, 3, 4}),
         "RoPE identity mismatch");
  ExpectThrows([&] { RoPE(q, k, 1, cos_table, sin_table); },
               "RoPE accepted a position outside its table");
}

void TestQuantization() {
  Tensor source({2, 32}, 0.0f);
  for (size_t i = 0; i < source.Size(); ++i) {
    source.data[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 4.0f;
  }
  std::vector<BlockQ80> q80 = QuantizeToQ80(source);
  Tensor restored80 = DeQuantizeFromQ80(q80, source.shape);
  for (size_t i = 0; i < source.Size(); ++i) {
    ExpectNear(restored80.data[i], source.data[i], 0.03f,
               "Q8_0 round-trip error is too large");
  }

  std::vector<BlockQ40> q40 = QuantizeToQ40(source);
  Tensor restored40 = DeQuantizeFromQ40(q40, source.shape);
  for (size_t i = 0; i < source.Size(); ++i) {
    ExpectNear(restored40.data[i], source.data[i], 0.3f,
               "Q4_0 round-trip error is too large");
  }

  Tensor input({32}, 1.0f);
  Tensor output = LinearQ80(input, q80, source.shape);
  Expect(output.shape == std::vector<int>({2}),
         "Q8_0 linear shape mismatch");
  float expected0 = std::accumulate(source.data.begin(), source.data.begin() + 32,
                                    0.0f);
  ExpectNear(output.data[0], expected0, 0.3f,
             "Q8_0 linear value mismatch");

  Tensor zero({64}, 0.0f);
  std::vector<BlockQ80> zero80 = QuantizeToQ80(zero);
  std::vector<BlockQ40> zero40 = QuantizeToQ40(zero);
  for (const BlockQ80& block : zero80) {
    Expect(block.d == 0, "zero Q8_0 block has a nonzero scale");
    for (int8_t value : block.qs) {
      Expect(value == 0, "zero Q8_0 block has uninitialized data");
    }
  }
  for (const BlockQ40& block : zero40) {
    Expect(block.d == 0, "zero Q4_0 block has a nonzero scale");
    for (uint8_t value : block.qs) {
      Expect(value == 0, "zero Q4_0 block has uninitialized data");
    }
  }

  Tensor q80_subnormal_probe({32}, 0.0f);
  q80_subnormal_probe.data[0] = 127.0f / 65536.0f;
  Expect(QuantizeToQ80(q80_subnormal_probe)[0].d == 0x0100,
         "Q8_0 subnormal scale was not converted to FP16");

  Tensor q40_subnormal_probe({32}, 0.0f);
  q40_subnormal_probe.data[0] = 7.0f / 65536.0f;
  Expect(QuantizeToQ40(q40_subnormal_probe)[0].d == 0x0100,
         "Q4_0 subnormal scale was not converted to FP16");

  Tensor q80_scale_probe({32}, 0.0f);
  q80_scale_probe.data[0] = 190.5f;  // max_abs / 127 = 1.5
  Expect(QuantizeToQ80(q80_scale_probe)[0].d == 0x3e00,
         "Q8_0 scale was not converted to the expected FP16 value");

  Tensor q40_scale_probe({32}, 0.0f);
  q40_scale_probe.data[0] = 10.5f;  // max_abs / 7 = 1.5
  Expect(QuantizeToQ40(q40_scale_probe)[0].d == 0x3e00,
         "Q4_0 scale was not converted to the expected FP16 value");
}

void TestBatchAndKvCache() {
  MiniBatch single = MiniBatch::GetSingleMiniBatch(7, 4);
  Expect(single.tokens == std::vector<int>({7}) &&
             single.positions == std::vector<int>({4}),
         "single MiniBatch mismatch");
  MiniBatch batch = MiniBatch::GetMultiMiniBatch({3, 4, 5}, 9);
  Expect(batch.positions == std::vector<int>({9, 10, 11}),
         "multi MiniBatch positions mismatch");

  KVCache cache(1, 3, 2, 2);
  Tensor key({2, 2}, 0.0f);
  key.data = {1, 2, 3, 4};
  Tensor value({2, 2}, 0.0f);
  value.data = {5, 6, 7, 8};
  cache.Write(0, 1, key, value);
  ExpectNear(cache.GetKeyPtr(0, 1, 1)[0], 3.0f, 0.0f,
             "KV key write mismatch");
  ExpectNear(cache.GetValuePtr(0, 1, 0)[1], 6.0f, 0.0f,
             "KV value write mismatch");
  ExpectThrows([&] { cache.Write(0, 3, key, value); },
               "KV cache accepted an out-of-range position");
}

void TestSampler() {
  Tensor logits({5}, 0.0f);
  logits.data = {1, 5, 3, 5, 2};
  Expect(MiniSampler::SampleGreedy(logits) == 1,
         "greedy sampler tie handling mismatch");

  SamplerParams params;
  params.temperature = 0.8f;
  params.top_k = 2;
  params.seed = 123;
  MiniSampler first(params);
  MiniSampler second(params);
  for (int i = 0; i < 20; ++i) {
    int a = first.Sample(logits, params);
    int b = second.Sample(logits, params);
    Expect(a == b, "fixed seed sampler is not reproducible");
    Expect(a == 1 || a == 3, "top-k sampler selected outside top two");
  }
  params.temperature = -1.0f;
  ExpectThrows([&] { (void)first.Sample(logits, params); },
               "sampler accepted negative temperature");
}

void TestChatPromptAndRadix() {
  ChatSession session;
  session.AddMessage("system", "rules");
  session.AddMessage("user", "hello");
  session.Record(2, 3, 4.5);
  Expect(session.messages_.size() == 2, "chat messages were not recorded");
  Expect(session.total_prompt_tokens_ == 2 &&
             session.total_generated_tokens_ == 3,
         "chat statistics mismatch");

  PromptBuilder builder;
  std::string prompt = builder.Build(session.messages_);
  Expect(prompt == "System: rules\nUser: hello\nAssistant: ",
         "plain prompt mismatch");
  builder.SetChatTemplate("qwen2");
  std::string qwen = builder.Build({{"user", "hello"}});
  Expect(qwen.find("<|im_start|>assistant\n") != std::string::npos,
         "Qwen generation prompt missing");

  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Insert({1, 2, 4});
  Expect(tree.Size() == 2, "radix tree size mismatch");
  Expect(tree.GetLongestPrefix({1, 2, 3, 8}) == 3,
         "radix tree prefix mismatch");
  tree.Clear();
  Expect(tree.Empty(), "radix tree clear mismatch");
}

void TestRequestAndBackend() {
  RequestContext request = StartRequest("test", "cpu", "models/tiny");
  request.prompt_tokens = 2;
  request.RecordEvent("stage", 1.25, 2, "detail");
  request.Finish();
  Expect(request.IsOk(), "new request is unexpectedly failed");
  Expect(FormatRequestTraceSummary(request).find("mode=test") !=
             std::string::npos,
         "request summary missing mode");
  Expect(FormatRequestTraceEvents(request).size() == 1,
         "request event missing");

  BackendConfig cpu;
  ValidateBackend(cpu);
  Expect(!CudaBackendBuilt(), "CPU build unexpectedly reports CUDA support");
  cpu.device_id_set = true;
  ExpectThrows([&] { ValidateBackend(cpu); },
               "CPU backend accepted a CUDA device id");
}

void TestThreadedMatmul() {
  Tensor a({5, 7}, 0.0f);
  Tensor b({7, 3}, 0.0f);
  for (size_t i = 0; i < a.Size(); ++i) {
    a.data[i] = static_cast<float>(i % 9) / 5.0f;
  }
  for (size_t i = 0; i < b.Size(); ++i) {
    b.data[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 4.0f;
  }
  Tensor expected = MatulDispatch(a, b, MatmulMode::kNaive);
  SetThreadCount(3);
  Tensor threaded = MatulDispatch(a, b, MatmulMode::kThread);
  Expect(threaded.shape == expected.shape, "threaded matmul shape mismatch");
  for (size_t i = 0; i < expected.Size(); ++i) {
    ExpectNear(threaded.data[i], expected.data[i], 1e-6f,
               "threaded matmul value mismatch");
  }
  SetThreadCount(0);
}

void TestTokenizerAndManifest() {
  AsciiTokenizer ascii;
  std::vector<int> ids = ascii.Encode("hello");
  Expect(ascii.Decode(ids) == "<bos>hello",
         "ASCII tokenizer round-trip mismatch");

  JsonTokenizer json("models/tiny/vocab.json");
  ids = json.Encode("hello!");
  Expect(json.Decode(ids) == "<bos>hello!",
         "JSON tokenizer round-trip mismatch");
  Expect(json.GetBOSId() == 1 && json.GetEOSId() == 2 &&
             json.GetUNKId() == 0,
         "JSON tokenizer special ids mismatch");

  ModelManifest manifest = ParseManifest("models/tiny/model.json");
  Expect(manifest.config.vocab_size == 128 && manifest.config.n_layers == 2,
         "manifest config mismatch");
  Expect(!manifest.tensor_infos.empty(), "manifest tensor list is empty");
  Expect(manifest.tokenizer.type == "json_vocab",
         "manifest tokenizer type mismatch");
}

using Test = std::pair<const char*, std::function<void()>>;

}  // namespace

int main() {
  const std::vector<Test> tests = {
      {"tensor", TestTensor},
      {"matmul_and_linear", TestMatmulAndLinear},
      {"elementwise_1d", TestElementwiseOneDimensional},
      {"elementwise_multidimensional", TestElementwiseMultiDimensional},
      {"normalization_and_rope", TestNormalizationAndRope},
      {"quantization", TestQuantization},
      {"batch_and_kv_cache", TestBatchAndKvCache},
      {"sampler", TestSampler},
      {"chat_prompt_and_radix", TestChatPromptAndRadix},
      {"request_and_backend", TestRequestAndBackend},
      {"threaded_matmul", TestThreadedMatmul},
      {"tokenizer_and_manifest", TestTokenizerAndManifest},
  };

  int failed = 0;
  for (const Test& test : tests) {
    try {
      test.second();
      std::cout << "[PASS] " << test.first << '\n';
    } catch (const std::exception& error) {
      ++failed;
      std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
    }
  }
  std::cout << "core_cpu_test: " << (tests.size() - failed) << " passed, "
            << failed << " failed\n";
  return failed == 0 ? 0 : 1;
}
