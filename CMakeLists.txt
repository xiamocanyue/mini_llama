cmake_minimum_required(VERSION 3.17)
project(mini-llama-cpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

set(MINI_LLAMA_CPU_SOURCES
    src/tensor.cpp
    src/model.cpp
    src/ops.cpp
    src/ascii_tokenizer.cpp
    src/json_tokenizer.cpp
    src/bpe_tokenizer.cpp
    src/gguf_tokenizer.cpp
    src/kv_cache.cpp
    src/context.cpp
    src/forward.cpp
    src/sampler.cpp
    src/loader.cpp
    src/chat.cpp
    src/prompt_builder.cpp
    src/terminal.cpp
    src/batch.cpp
    src/debug.cpp
    src/gguf.cpp
    src/gguf_loader.cpp
    src/quant.cpp
    src/quantized_tensor.cpp
    src/thread_pool.cpp
    src/matmul_optimized.cpp
    src/request_context.cpp
    src/radix_tree.cpp
    test/main_cpu_chat_test.cpp
)

add_executable(mini-llama-cpu ${MINI_LLAMA_CPU_SOURCES})
set_target_properties(mini-llama-cpu PROPERTIES OUTPUT_NAME mini-llama-cpu)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
