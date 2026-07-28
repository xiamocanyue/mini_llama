#include "include/chat.h"

namespace mini_llama {

void ChatSession::Clear() {
    messages_.clear();
    token_history_.clear();
    prefix_cache_.Clear();
    total_generated_tokens_ = 0;
    total_prompt_tokens_ = 0;
    total_time_ms_ = 0.0;
}

void ChatSession::AddMessage(const std::string& role, const std::string& content) {
    messages_.push_back({role, content});
}

void ChatSession::Record(int prompt_tokens_num, int generated_tokens_num, double time_ms) {
    total_prompt_tokens_ += prompt_tokens_num;
    total_generated_tokens_ += generated_tokens_num;
    total_time_ms_ += time_ms;
}
void ChatSession::SetTokenHistory(const std::vector<int>& tokens) {
    token_history_ = tokens;
}
void ChatSession::AppendToken(int token) {
    token_history_.push_back(token);
}
void ChatSession::RecordPrefix(const std::vector<int>& tokens) {
    prefix_cache_.Insert(tokens);
}
size_t ChatSession::GetLongestCachePrefix(const std::vector<int>& tokens) {
    return prefix_cache_.GetLongestPrefix(tokens);
}



}