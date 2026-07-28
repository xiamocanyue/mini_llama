#pragma once 

#include "include/context.h"
#include "include/radix_tree.h"
#include "include/sampler.h"

namespace mini_llama {

struct ChatMessage {
    std::string role;  //system, user, assistant
    std::string content;
};

class ChatSession {
public:
    ChatSession() = default;

    void Clear();
    void AddMessage(const std::string& role, const std::string& content);
    void Record(int prompt_tokens_num, int generated_tokens_num, double time_ms);
    void SetTokenHistory(const std::vector<int>& tokens);
    void AppendToken(int token);
    void RecordPrefix(const std::vector<int>& tokens);
    size_t GetLongestCachePrefix(const std::vector<int>& tokens);


public:
    std::vector<ChatMessage> messages_;
    //这里的token_history，是整个会话的记录，不随ctx的重建而销毁。
    //每次用message_编码后的token作为基准，再decode的时候逐词添加。
    std::vector<int> token_history_;
    RadixTree prefix_cache_;
    SamplerParams sampler_params_;

    //记录的是整体prompt，包括公共前缀的长度
    int total_prompt_tokens_ = 0;
    int total_generated_tokens_ = 0;
    double total_time_ms_ = 0.0;
};


}