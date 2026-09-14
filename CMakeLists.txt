cmake_minimum_required(VERSION 3.18)
project(m_mini_llam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MINI_LLAMA_CUDA "Build the NVIDIA CUDA backend" OFF)
option(MINI_LLAMA_BUILD_TESTS "Build CPU tests" ON)

find_package(Threads REQUIRED)

if(MINI_LLAMA_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
endif()

set(MINI_LLAMA_CORE_SOURCES
    src/ascii_tokenizer.cpp
    src/backend.cpp
    src/batch.cpp
    src/json_tokenizer.cpp
    src/bpe_tokenizer.cpp
    src/chat.cpp
    src/context.cpp
    src/cuda_attention.cpp
    src/cuda_data.cpp
    src/cuda_kv_cahce.cpp
    src/cuda_matmul.cpp
    src/cuda_ops.cpp
    src/cuda_quant.cpp
    src/cuda_tensor.cpp
    src/debug.cpp
    src/forward.cpp
    src/gguf.cpp
    src/gguf_loader.cpp
    src/gguf_tokenizer.cpp
    src/kv_cache.cpp
    src/loader.cpp
    src/matmul_optimized.cpp
    src/model.cpp
    src/ops.cpp
    src/prompt_builder.cpp
    src/quant.cpp
    src/quantized_tensor.cpp
    src/radix_tree.cpp
    src/request_context.cpp
    src/sampler.cpp
    src/tensor.cpp
    src/terminal.cpp
    src/thread_pool.cpp
)

if(MINI_LLAMA_CUDA)
    list(APPEND MINI_LLAMA_CORE_SOURCES
        src/cuda_attention.cu
        src/cuda_ops.cu
        src/cuda_quant.cu
    )
endif()

add_library(mini-llama-core STATIC ${MINI_LLAMA_CORE_SOURCES})
target_include_directories(mini-llama-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mini-llama-core PUBLIC Threads::Threads)

if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-core PUBLIC MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-core PUBLIC CUDA::cudart CUDA::cublas)
    set_target_properties(mini-llama-core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(mini-llama src/main.cpp)
target_link_libraries(mini-llama PRIVATE mini-llama-core)

if(MINI_LLAMA_BUILD_TESTS)
    enable_testing()

    function(add_mini_llama_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE mini-llama-core)
        add_test(NAME ${target} COMMAND ${target} ${ARGN})
        set_tests_properties(${target} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endfunction()

    add_mini_llama_test(json-tokenizer-test test/json_tokenizer_test.cpp)
    add_mini_llama_test(core-cpu-test test/core_cpu_test.cpp)
    add_mini_llama_test(kv-cache-test test/kv_cache_test.cpp)
    add_mini_llama_test(transformer-forward-test test/transformer_forward_test.cpp)
    add_mini_llama_test(cpu-generate-tiny-test test/main_cpu_generate_test.cpp tiny)
    add_mini_llama_test(cpu-chat-tiny-test test/main_cpu_chat_test.cpp tiny)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME python-model-alignment-test
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_model_alignment.py
                    --binary $<TARGET_FILE:mini-llama>
        )
        set_tests_properties(python-model-alignment-test PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

    if(MINI_LLAMA_CUDA)
        add_mini_llama_test(gpu-core-test test/gpu_core_test.cpp)
        set_tests_properties(gpu-core-test PROPERTIES LABELS "gpu")

        if(Python3_Interpreter_FOUND)
            add_test(
                NAME gpu-model-alignment-test
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gpu_alignment.py
                        --binary $<TARGET_FILE:mini-llama>
            )
            set_tests_properties(gpu-model-alignment-test PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                LABELS "gpu"
            )
        endif()
    endif()
endif()
