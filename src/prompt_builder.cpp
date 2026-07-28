#include "include/prompt_builder.h"

#include <cstring>
#include <stack>

namespace mini_llama {

namespace {

struct EvalContext {
    const std::vector<ChatMessage>* total_messages_ptr = nullptr;
    const ChatMessage* current_message_ptr = nullptr;
    size_t current_message_index = 0;
    bool is_add_generation_prompt = false;
};

//删除首尾空格
std::string Trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<char>(s[start]))) {
        start++;
    }
    size_t end = s.size() - 1;
    while (end > start && std::isspace(static_cast<char>(s[end]))) {
        end--;
    }

    return s.substr(start, end - start + 1);
}

//删除Jinja中左右两边的减号并去除空格
std::string StripAndTrimJinjaMarkers(const std::string& s) {
    std::string value = Trim(s);
    if (!value.empty() && value.front() == '-') {
        value = Trim(value.substr(1));
    }
    if (!value.empty() && value.back() == '-') {
        value.pop_back();
        value = Trim(value);
    }
    return value;
}

void SkipSpace(size_t& pos, const std::string& s) { 
    while (pos < s.size() && std::isspace(static_cast<char>(s[pos]))) {
        pos++;
    }
}

/*
处理Jinja 拼接消息，处理
加号,add_generation_prompt, message['role'],message['content'],loop.first, 
message[0]['role'], message[0]['content']  
*/   
std::string EvalExpr(const std::string& expr_s, const EvalContext& ctx) {
    std::string expr = Trim(expr_s);
    if (expr.empty()) {
        return "";
    }

    size_t pos = 0;
    std::vector<std::string> parts;
    while (pos < expr.size()) {
        SkipSpace(pos, expr);
        if (pos >= expr.size()) {
            return "";
        }
        // 处理像'role'的单引号，收集单词 
        if (expr[pos] == '\'') {
            pos++;
            std::string word;
            while (pos < expr.size() && expr[pos] != '\'') {
                word += expr[pos];
                pos++;
            }
            if (pos < expr.size() && expr[pos] == '\'') {
                pos++;
            }
            parts.push_back(word);     
        } else { //将[]中间，和.之后的内容直接读取，不用看有没有引号和其他特殊符号
            size_t start = pos;
            while (pos < expr.size() && expr[pos] != '+' && expr[pos] != '\'') {
                if (expr[pos] == '[') {
                    pos++;
                    while (pos < expr.size() && expr[pos] != ']') {
                        pos++;
                    }
                    if (pos < expr.size()) {
                        pos++;
                    }
                } else if (expr[pos] == '.') {
                    pos++;
                    while (pos < expr.size() && expr[pos] != '+' && expr[pos] != '\'' 
                            && expr[pos] != '[' && expr[pos] != '.' 
                            && !std::isspace(static_cast<char>(expr[pos]))) {
                        pos++;
                    }
                } else {
                    pos++;
                }
            }

            std::string token = Trim(expr.substr(start, pos - start));
            if (!token.empty()) {
                if (token == "add_generation_prompt") {
                    parts.push_back(ctx.is_add_generation_prompt ? "true" : "false");
                } else if (token == "message['role']" || token == "message[\"role\"]") {
                    parts.push_back(ctx.current_message_ptr ? ctx.current_message_ptr->role : "");
                } else if (token == "message['content']" || token == "message[\"content\"]") {
                    parts.push_back(ctx.current_message_ptr ? ctx.current_message_ptr->content : "");
                } else if (token == "loop.first") {
                    parts.push_back(ctx.current_message_index == 0 ? "true" : "false");
                } else if (token.find("messages[") == 0) {  // 处理 messages[0]['role']情况
                    size_t end = token.find(']');
                    if (end != std::string::npos && end + 1 < token.size() && token[end + 1] == '[') {
                        size_t begin = static_cast<size_t>(std::strlen("messages["));
                        std::string message_index_str = Trim(token.substr(begin, end - begin));
                        
                        std::string key;
                        size_t key_start = end + 2;
                        size_t key_end = token.find(']', key_start);
                        if (key_end != std::string::npos) {
                            key = token.substr(key_start, key_end - key_start);
                            if (key.front() == '\'' && key.back() == '\'' ||
                                key.front() == '\"' && key.back() == '\"') {
                                key = key.substr(1, key.size() - 2);            
                            }
                        }
                        //取出了消息下标index， 和消息内容key,进行处理
                        try {
                            int index = std::stoi(message_index_str);
                            if (ctx.total_messages_ptr && index >= 0 && index < static_cast<int>(ctx.total_messages_ptr->size())) {
                                const auto& messages = *(ctx.total_messages_ptr);
                                const ChatMessage& msg = messages[index];
                                if (key == "role") {
                                    parts.push_back(msg.role);
                                } else if (key == "content") {
                                    parts.push_back(msg.content);
                                } else {
                                    parts.push_back("");
                                }
                            } else {
                                parts.push_back("");
                            }
                        } catch (...) {
                            parts.push_back("");
                        }
                    } else {
                        parts.push_back("");
                    }
                } else{
                    parts.push_back("");
                }
            }

        }
        // 前面处理了'', 又处理了[], 最后处理 +号
        SkipSpace(pos, expr);
        if (pos < expr.size() && expr[pos] == '+') {
            pos++;
        }
    }

    //汇总结果
    std::string result;
    for (auto& p : parts) {
        result  += p;
    }
    return result;
}

// 处理if判断中的条件
bool EvalCondition(const std::string& cond_s, const EvalContext& ctx) {
    std::string cond = Trim(cond_s);
    if (cond.empty()) {
        return false;
    }

    //处理and
    size_t and_pos = cond.find(" and ");
    while (and_pos != std::string::npos) {
        std::string left = Trim(cond.substr(0, and_pos));
        std::string right = Trim(cond.substr(and_pos + 5));
        return EvalCondition(left, ctx) && EvalCondition(right, ctx);
    }

    //处理 or
    size_t or_pos = cond.find(" or ");
    while (or_pos != std::string::npos) {
        std::string left = Trim(cond.substr(0, or_pos));
        std::string right = Trim(cond.substr(or_pos + 4));
        return EvalCondition(left, ctx) || EvalCondition(right, ctx);
    }

    //处理 !=
    size_t neq_pos = cond.find(" != ");
    if (neq_pos != std::string::npos) {
        std::string left = EvalExpr(cond.substr(0, neq_pos), ctx);
        std::string right = EvalExpr(cond.substr(neq_pos + 4), ctx);
        return left != right;
    }

    //处理 ==
    size_t eq_pos = cond.find(" == ");
    if (eq_pos != std::string::npos) {
        std::string left = EvalExpr(cond.substr(0, eq_pos), ctx);
        std::string right = EvalExpr(cond.substr(eq_pos + 4), ctx);
        return left == right;
    }

    //处理单条判断
    std::string val = EvalExpr(cond, ctx);
    if (val.empty()) {
        return false;
    }
    if (val == "true" || val == "True") {
        return true;
    }
    if (val == "false" || val == "False") {
        return false;
    }
    return true;
}

struct Instr {
    enum Type {
        kText,
        kOutput,
        kIf,
        kEndIf,
        kFor,
        kEndFor
    } type;
    std::string text;
    size_t jump = 0;
};

struct ForFrame {
    size_t pc = 0;
    size_t index = 0;
};

// 将for if 转换成一个个的 Instr
/*
    {% for message in messages %}
    {{ message['role'] }}
    {% endfor %}

    处理成

    kFor
    kOutput
    kEndFor
*/
std::vector<Instr> ComplieTemplate(const std::string& input) {
    std::vector<Instr> instrs;
    std::stack<size_t> if_stack;
    std::stack<size_t> for_stack;

    size_t pos = 0;
    while (pos < input.size()) {
        // 处理具有for或者if的语句
        if (pos + 1 < input.size() && input[pos] == '{' && input[pos] == '%') {
            pos += 2;
            size_t end = input.find("%}");
            if (end == std::string::npos) {
                break;
            }

            std::string sentence = StripAndTrimJinjaMarkers(input.substr(pos, end - pos));
            pos = end + 2;
            
            //开头是for
            if (sentence.rfind("for", 0) == 0) {
                instrs.push_back({Instr::kFor, sentence});
                for_stack.push(instrs.size() - 1);   //用size计算下标
            } else if (sentence == "endfor") {
                if (for_stack.empty()) {
                    continue;
                }

                size_t for_pc = for_stack.top();
                for_stack.pop();
                instrs.push_back({Instr::kEndFor, "", for_pc});
                instrs[for_pc].jump = instrs.size();
            } else if (sentence.rfind("if", 0) == 0) { 
            //开头是 if
                instrs.push_back({Instr::kIf, sentence.substr(3)});
                if_stack.push(instrs.size() - 1);
            } else if (sentence == "endif") {
                if (if_stack.empty()) {
                    continue;
                }
                size_t if_pc = if_stack.top();
                if_stack.pop();
                instrs.push_back({Instr::kEndIf});
                instrs[if_pc].jump = instrs.size();
            }
        //处理没有for或if的语句
        } else if (pos + 1 < input.size() && input[pos] == '{' && input[pos + 1] == '{') {
            pos += 2;
            size_t end = input.find("}}", pos);
            if (end == std::string::npos) {
                break;
            }
            std::string token = StripAndTrimJinjaMarkers(input.substr(pos, end - pos));
            pos = end + 2;
            instrs.push_back({Instr::kOutput, token});
        } else {  
        //处理普通的文本
            size_t start = pos;
            while (pos < input.size() && 
                    !(input[pos] == '{' && pos + 1 < input.size() && 
                    (input[pos + 1] == '%' || input[pos] == '{'))) {
                pos++;
            }
            std::string text = input.substr(start, pos - start);
            if (!text.empty()) {
                instrs.push_back({Instr::kText, text});
            }
        }
    }

    while (!if_stack.empty()) {
        instrs[if_stack.top()].jump = instrs.size();
        if_stack.pop();
    }
    while (!for_stack.empty()) {
        instrs[for_stack.top()].jump = instrs.size();
        for_stack.pop();
    }

    return instrs;
}

std::string ExecuteTemplate(const std::vector<Instr>& instrs, 
                            const std::vector<ChatMessage>& messages, 
                            bool is_add_gen) {
    std::string output;
    EvalContext ctx;
    ctx.total_messages_ptr = &messages;
    ctx.is_add_generation_prompt = is_add_gen;
    std::stack<ForFrame> for_stack;
    size_t pc = 0;

    while (pc < instrs.size()) {
        const Instr& instr = instrs[pc];
        switch (instr.type) {
            case Instr::kText :
                output += instr.text;
                pc++;
                break;
            case Instr::kOutput :
                output += EvalExpr(instr.text, ctx);
                pc++;
                break;
            case Instr::kIf :
               if (!EvalCondition(instr.text, ctx)) {
                    pc = instr.jump;
               } else {
                    pc++;
               }
               break;
            case Instr::kEndIf :
               pc++;
               break;
            case Instr::kFor: {
                if (for_stack.empty() || for_stack.top().pc != pc) {
                    ForFrame frame;
                    frame.pc = pc;
                    frame.index = 0;
                    for_stack.push(frame);
                    if (messages.empty()) {
                        for_stack.pop();
                        pc = instr.jump;
                    } else {
                        ctx.current_message_index = 0;
                        ctx.current_message_ptr = &messages[0];
                        pc++;
                    }
                } else {
                    ForFrame& frame = for_stack.top();
                    frame.index++;
                    if (frame.index >= messages.size()) {
                        for_stack.pop();
                        pc = instr.jump;
                    } else {
                        ctx.current_message_index = frame.index;
                        ctx.current_message_ptr = &messages[frame.index];
                        pc++;
                    }
                }
                break;
            }
            case Instr::kEndFor:
                pc = instr.jump;
                break;
        }        
    }
    return output;
}

} // namespace 

void PromptBuilder::SetChatTemplate(const std::string& template_str) {
    chat_template_ = template_str;
}

std::string PromptBuilder::BuildPlain(const std::vector<ChatMessage>& messages) const {
    std::string prompt;
    for (auto& msg : messages) {
        if (msg.role == "System" || msg.role == "system") {
            prompt += "System: " + msg.content + '\n';
        } else if (msg.role == "User" || msg.role == "user") {
            prompt += "User: " + msg.content + '\n';
        } else if (msg.role == "Assistant" || msg.role == "assistant") {
            prompt += "Assistant: " + msg.content + '\n';
        }
    }
    prompt += "Assistant: ";
    return prompt;
}

//message: <|im_start|>{role}\n{content}<|im_end|>\n
//结尾 <|im_start|>assistant\n
std::string PromptBuilder::BuildeQwen(const std::vector<ChatMessage>& messages) const {
    std::string prompt;
    bool is_have_system = false;
    for (auto& msg : messages) {
        if (msg.role == "system") {
            is_have_system = true;
            break;
        }
    }
    if (!is_have_system) {
        prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    }

    for (auto& msg : messages) {
        if (msg.role != "system" && msg.role != "user" && msg.role != "assistant") {
            continue;
        }
        prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n"; 
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}


std::string PromptBuilder::Build(const std::vector<ChatMessage>& messages) const {
    if (chat_template_.empty()) {
        return BuildPlain(messages);
    }
    if (chat_template_ == "qwen2") {
        return BuildeQwen(messages);
    }
    if (chat_template_.size() >= 2 && chat_template_[0] == '{' && 
        (chat_template_[1] == '%' || chat_template_[1] == '{')) {
        auto instrs = ComplieTemplate(chat_template_);
        return ExecuteTemplate(instrs, messages, true);
    }

    return BuildPlain(messages);
}




}