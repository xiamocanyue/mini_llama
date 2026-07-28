
static int RunChat(int argc, char** argv) {
  // if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
  //                   std::strcmp(argv[2], "--help") == 0)) {
  //   PrintRunUsage(argv[0]);
  //   return 0;
  // }

  // if (argc < 3) {
  //   std::cerr << "Missing model directory.\n";
  //   PrintRunUsage(argv[0]);
  //   return 1;
  // }

  // std::string model_dir = argv[2];
  // float temperature = 0.0f;
  // int top_k = 0;
  // unsigned int seed = 0;
  // int max_response_tokens = 64;
  // std::string tokenizer_path;
  // BackendConfig backend_config;

  // for (int i = 3; i < argc; ++i) {
  //   if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
  //     if (!ParseFloatArg(argv[++i], temperature)) {
  //       std::cerr
  //           << "Invalid --temperature value. Expected a non-negative float.\n";
  //       return 1;
  //     }
  //   } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
  //     if (!ParseIntArg(argv[++i], top_k)) {
  //       std::cerr
  //           << "Invalid --top-k value. Expected a non-negative integer.\n";
  //       return 1;
  //     }
  //   } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
  //     if (!ParseUintArg(argv[++i], seed)) {
  //       std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
  //       return 1;
  //     }
  //   } else if ((std::strcmp(argv[i], "-n") == 0 ||
  //               std::strcmp(argv[i], "--n-predict") == 0) &&
  //              i + 1 < argc) {
  //     if (!ParseIntArg(argv[++i], max_response_tokens) ||
  //         max_response_tokens <= 0) {
  //       std::cerr
  //           << "Invalid --n-predict value. Expected a positive integer.\n";
  //       return 1;
  //     }
  //   } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
  //     tokenizer_path = argv[++i];
  //   } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
  //     if (!ParseBackendArg(argv[++i], backend_config)) {
  //       std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
  //       return 1;
  //     }
  //   } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
  //     if (!ParseDeviceArg(argv[++i], backend_config)) {
  //       std::cerr
  //           << "Invalid --device value. Expected a non-negative integer.\n";
  //       return 1;
  //     }
  //   } else if (std::strcmp(argv[i], "-h") == 0 ||
  //              std::strcmp(argv[i], "--help") == 0) {
  //     PrintRunUsage(argv[0]);
  //     return 0;
  //   } else {
  //     std::cerr << "Unknown argument: " << argv[i] << "\n";
  //     PrintRunUsage(argv[0]);
  //     return 1;
  //   }
  // }

  // if (!ValidateBackendOrPrint(backend_config)) {
  //   return 1;
  // }

  // Load model
  LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
  // if (!lm.model.loaded) {
  //   std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
  //   return 1;
  // }
  // if (!lm.tokenizer) {
  //   std::cerr << "Failed to load tokenizer.\n";
  //   return 1;
  // }
  // if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
  //   std::cerr << "Model vocab_size must be at least "
  //             << lm.tokenizer->vocab_size() << " for the tokenizer.\n";
  //   return 1;
  // }

  MiniLlamaModel& model = lm.model;
  // if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
  //   return 1;
  // }
  std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

  PromptBuilder builder;
  // if (!lm.chat_template.empty()) {
  //   builder.SetChatTemplate(lm.chat_template);
  // }
  Terminal term;
  ChatSession session;
  session.sampling_params.temperature = temperature;
  session.sampling_params.top_k = top_k;
  session.sampling_params.seed = seed;

  // Plain text mode keeps the default system message in session state.
  // if (lm.chat_template.empty()) {
  //   session.AddMessage("system", "You are a helpful assistant.");
  // }

  // term.PrintMessage("mini-llama.cpp chat");
  // term.PrintMessage("backend: " + BackendKindName(backend_config.kind));
  // if (backend_config.kind == BackendKind::kCuda) {
  //   LinearWeightTypeCounts counts = ModelLinearWeightTypeCounts(model);
  //   size_t uploaded_f32 = ModelCudaUploadedF32WeightCount(model);
  //   size_t uploaded_q8_0 = ModelCudaUploadedQ80WeightCount(model);
  //   size_t uploaded_q4_0 = ModelCudaUploadedQ40WeightCount(model);
  //   size_t uploaded_q4_1 = ModelCudaUploadedQ41WeightCount(model);
  //   size_t q8_0_fallback =
  //       counts.q8_0 >= uploaded_q8_0 ? counts.q8_0 - uploaded_q8_0 : 0;
  //   size_t q4_0_fallback =
  //       counts.q4_0 >= uploaded_q4_0 ? counts.q4_0 - uploaded_q4_0 : 0;
  //   size_t q4_1_fallback =
  //       counts.q4_1 >= uploaded_q4_1 ? counts.q4_1 - uploaded_q4_1 : 0;
  //   term.PrintMessage("cuda: " + CudaDeviceSummary(backend_config.device_id));
  //   term.PrintMessage("uploaded weights: " +
  //                     std::to_string(ModelCudaUploadedWeightCount(model)));
  //   term.PrintMessage(
  //       "gpu memory used: " + std::to_string(ModelCudaMemoryBytes(model)) +
  //       " bytes (" + FormatMb(ModelCudaMemoryBytes(model)) + ")");
  //   term.PrintMessage(
  //       "cuda f32 Linear weights: " + std::to_string(uploaded_f32) + "/" +
  //       std::to_string(counts.total()));
  //   if (counts.q8_0 > 0) {
  //     term.PrintMessage(
  //         "cuda q8_0 Linear weights: " + std::to_string(uploaded_q8_0) + "/" +
  //         std::to_string(counts.q8_0));
  //   }
  //   if (counts.q4_0 > 0) {
  //     term.PrintMessage(
  //         "cuda q4_0 Linear weights: " + std::to_string(uploaded_q4_0) + "/" +
  //         std::to_string(counts.q4_0));
  //   }
  //   if (counts.q4_1 > 0) {
  //     term.PrintMessage(
  //         "cuda q4_1 Linear weights: " + std::to_string(uploaded_q4_1) + "/" +
  //         std::to_string(counts.q4_1));
  //   }
  //   if (q8_0_fallback + q4_0_fallback + q4_1_fallback > 0) {
  //     term.PrintMessage(
  //         "cuda fallback: unsupported or missing quantized Linear weights use "
  //         "CPU path" +
  //         std::string(" (Q8_0=") + std::to_string(q8_0_fallback) +
  //         ", Q4_0=" + std::to_string(q4_0_fallback) +
  //         ", Q4_1=" + std::to_string(q4_1_fallback) + ")");
  //   }
  // }
  // term.PrintMessage(BackendExecutionNote(backend_config));
  // term.PrintMessage("Type /help for commands, /exit to quit.\n");
  // if (lm.chat_template.empty() && model.config.max_seq_len <= 256) {
  //   term.PrintMessage(
  //       "Tiny teaching model: random weights, small context window, smoke-test "
  //       "output.");
  //   term.PrintMessage(
  //       "Real chat demo: ./build/mini-llama run models/chat -n 8\n");
  // }

  MiniLlamaContext ctx(&model);

  while (true) {
    term.PrintUserPrompt();
    std::string input = term.ReadLine();
    // if (input.empty() && std::cin.eof()) {
    //   break;
    // }

    // Handle commands
    // if (input == "/help") {
    //   term.PrintHelp();
    //   continue;
    // }
    // if (input == "/exit") {
    //   break;
    // }
    // if (input == "/clear") {
    //   session.Clear();
    //   ctx = MiniLlamaContext(&model);
    //   if (lm.chat_template.empty()) {
    //     session.AddMessage("system", "You are a helpful assistant.");
    //   }
    //   term.PrintMessage("Chat history cleared.\n");
    //   continue;
    // }
    // if (input == "/stats") {
    //   term.PrintStats(session);
    //   continue;
    // }
    // if (input == "/params") {
    //   term.PrintParams(session.sampling_params);
    //   continue;
    // }
    // if (!input.empty() && input[0] == '/') {
    //   term.PrintMessage("Unknown command: " + input + "\n");
    //   continue;
    // }
    // if (input.empty()) {
    //   continue;
    // }

    // RequestContext request =
    //     StartRequest("run", BackendKindName(backend_config.kind), model_dir);

    std::vector<ChatMessage> candidate_messages = session.messages;
    candidate_messages.push_back({"user", input});

    // auto stage_start = RequestClock::now();
    std::string prompt_text = builder.Build(candidate_messages);
    // request.RecordEvent("prompt_build", ElapsedMs(stage_start), 0,
    //                     "messages=" +
    //                         std::to_string(candidate_messages.size()));
    // stage_start = RequestClock::now();
    std::vector<int> tokens = tokenizer->Encode(prompt_text);
    // request.tokenize_ms = ElapsedMs(stage_start);
    // request.prompt_tokens = static_cast<int>(tokens.size());
    // request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
    //                     "prompt");

    // if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
    //   request.SetError("prompt uses " + std::to_string(tokens.size()) +
    //                    " tokens, context window is " +
    //                    std::to_string(model.config.max_seq_len) +
    //                    ". Use /clear or a shorter prompt.");
    //   request.Finish();
    //   PrintRequestTrace(request);
    //   term.PrintMessage("Error: " + request.error + "\n");
    //   continue;
    // }

    // int max_response =
    //     model.config.max_seq_len - static_cast<int>(tokens.size());
    // if (max_response > max_response_tokens) {
    //   max_response = max_response_tokens;
    // }

    session.messages = candidate_messages;
    MiniSampler sampler(session.sampling_params);
    Tensor logits;

    // auto start = std::chrono::steady_clock::now();

    size_t cached_prefix_len = session.LongestCachedPrefix(tokens);
    size_t context_prefix_len = CommonPrefixLength(ctx.token_history, tokens);
    size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);
    // if (prefix_len >= tokens.size()) {
    //   ctx = MiniLlamaContext(&model);
    //   prefix_len = 0;
    // } else if (prefix_len == 0) {
    //   ctx = MiniLlamaContext(&model);
    // } else if (prefix_len < ctx.token_history.size()) {
    //   ctx.token_history.resize(prefix_len);
    //   ctx.pos = static_cast<int>(prefix_len - 1);
    // }
    std::vector<int> new_prompt_tokens(
        tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
    session.SetTokenHistory(tokens);

    // Prefill only the context suffix that is missing from KV cache.
    {
      MiniBatch prefill = MiniBatch::FromTokens(new_prompt_tokens,
                                                static_cast<int>(prefix_len));
      // stage_start = RequestClock::now();
      logits = ForwardBatch(ctx, model, prefill);
      // request.prefill_ms = ElapsedMs(stage_start);
      request.prefill_tokens = static_cast<int>(new_prompt_tokens.size());
      // request.RecordEvent(
      //     "prefill", request.prefill_ms, request.prefill_tokens,
      //     "radix_hit=" + std::to_string(cached_prefix_len) +
      //         ", prefix_reuse=" + std::to_string(prefix_len));
      ctx.n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());
    }

    // Generate response
    std::vector<int> generated_ids;
    std::string streamed_reply;
    int generated_count = 0;

    term.PrintAssistantPrefix();
    try {
      for (int i = 0; i < max_response; ++i) {
        // stage_start = RequestClock::now();
        int next_token = sampler.Sample(logits, session.sampling_params);
        // request.sample_ms += ElapsedMs(stage_start);
        tokens.push_back(next_token);
        session.AppendToken(next_token);
        ++generated_count;

        if (next_token != tokenizer->eos_id()) {
          generated_ids.push_back(next_token);
          std::string current_reply = tokenizer->Decode(generated_ids);
          if (current_reply.size() > streamed_reply.size()) {
            term.PrintTokenText(current_reply.substr(streamed_reply.size()));
            term.Flush();
            streamed_reply = current_reply;
          }
        }

        MiniBatch decode_batch =
            MiniBatch::Single(next_token, static_cast<int>(tokens.size() - 1));
        // stage_start = RequestClock::now();
        logits = ForwardBatch(ctx, model, decode_batch);
        // double decode_ms = ElapsedMs(stage_start);
        // request.decode_ms += decode_ms;
        // request.RecordEvent("decode", decode_ms, 1,
        //                     "pos=" + std::to_string(tokens.size() - 1));
        ++ctx.n_decode_tokens;
        ++request.decode_tokens;
        if (next_token == tokenizer->eos_id()) {
          break;
        }
      }
    // } catch (const std::exception& e) {
    //   term.NewLine();
    //   // request.SetError("Inference error: " + std::string(e.what()));
    //   // request.Finish();
    //   // PrintRequestTrace(request);
    //   term.PrintMessage(request.error + "\n");
    //   continue;
    // }

    // Decode and output
    std::string assistant_reply = tokenizer->Decode(generated_ids);
    if (assistant_reply.size() > streamed_reply.size()) {
      term.PrintTokenText(assistant_reply.substr(streamed_reply.size()));
    }
    // term.NewLine();
    // term.NewLine();

    // auto end = std::chrono::steady_clock::now();
    // double elapsed_ms =
    //     std::chrono::duration<double, std::milli>(end - start).count();

    session.AddMessage("assistant", assistant_reply);
    session.RecordTurn(static_cast<int>(new_prompt_tokens.size()),
                       generated_count, elapsed_ms);
    // request.generated_tokens = generated_count;
    // request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
    //                     "generated_tokens");
    session.RecordPrefix(session.token_history);
    // request.Finish();
    PrintRequestTrace(request);
  }

  term.PrintMessage("Goodbye.\n");
  return 0;
}