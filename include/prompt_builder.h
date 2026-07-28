#pragma once 

#include <vector>
#include <string>

#include "include/chat.h"

namespace mini_llama {

class PromptBuilder {
public:
    PromptBuilder() = default;
    void SetChatTemplate(const std::string& template_str);

    std::string Build(const std::vector<ChatMessage>& messages) const;


private:
    std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
    std::string BuildeQwen(const std::vector<ChatMessage>& messages) const;

private:
    std::string chat_template_;
};

std::string LoadChatTemplateFromGGUF(const std::string& gguf_path);

}