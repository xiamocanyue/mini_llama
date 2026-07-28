#pragma once 

#include "include/model.h"

namespace mini_llama {

MiniLlamaModel LoadGgufModel(const std::string& gguf_path);

}