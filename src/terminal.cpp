#include "include/terminal.h"

#include <iostream>
#include <iomanip>

namespace mini_llama {


void Terminal::PrintUserPrompt() const {
    std::cout << "mini_llama > ";
    std::cout.flush();
}

void Terminal::PrintAssistantPrefix() const{
    std::cout << "assistant > ";
    std::cout.flush();
}

void Terminal::PrintTokenText(const std::string& text) const {
    std::cout << text;
}

void Terminal::PrintHelp() const {
    std::cout << "Commands:\n"
              << "  /help    Show this help message\n"
              << "  /clear   Clear chat history and context\n"
              << "  /stats   Show session statistics\n"
              << "  /params  Show current sampling parameters\n"
              << "  /exit    Exit chat\n";
}

void Terminal::PrintStats(const ChatSession& session) const {
    std::cout << "Session stats:\n"
            << "  messages:           " << session.messages_.size() << "\n"
            << "  context tokens:     " << session.token_history_.size() << "\n"
            << "  total prompt tokens: " << session.total_prompt_tokens_ << "\n"
            << "  total generated:     " << session.total_generated_tokens_ << "\n"
            << "  total time:          " << std::fixed << std::setprecision(2)
            << session.total_time_ms_ << " ms\n"
            << "  tokens/s:            ";

    float token_speed = session.total_generated_tokens_ * 1000.0 / session.total_time_ms_;
    if (session.total_time_ms_ > 0.0) {
        std::cout << std::fixed << std::setprecision(2) << token_speed << '\n';
    } else {
        std::cout << "n/a" << '\n';
    }
}

void Terminal::PrintParams(const SamplerParams& params) const {
    std::cout << std::defaultfloat << "Sampling params:\n"
            << "  temperature: " << params.temperature << "\n"
            << "  top_k:       " << params.top_k << "\n"
            << "  seed:        " << params.seed << "\n";
}

void Terminal::PrintMessage(const std::string& msg) const {
    std::cout << msg << '\n';
}

std::string Terminal::ReadLine() const {
    std::string line;
    if (std::getline(std::cin, line)) {
        return line;
    }
    return "";
}

void Terminal::Flush() const {
    std::cout.flush();
}

void Terminal::NewLine() const {
    std::cout << '\n';
}

}