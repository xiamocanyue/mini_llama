#pragma once 


#include "include/chat.h"
#include "include/model.h"
#include "include/tokenizer.h"

namespace mini_llama {

class Terminal {
public:
    Terminal() = default;

    void PrintUserPrompt() const;

    void PrintAssistantPrefix() const;

    void PrintTokenText(const std::string& text) const;

    void PrintHelp() const;

    void PrintStats(const ChatSession& session) const;
    
    void PrintParams(const SamplerParams& params) const;

    void PrintMessage(const std::string& msg) const;

    std::string ReadLine() const;

    void Flush() const;

    void NewLine() const;


};



}