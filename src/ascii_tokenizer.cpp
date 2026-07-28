#include "include/tokenizer.h"

namespace mini_llama {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(GetBOSId());
    for (unsigned char c : text) {
        int id = static_cast<int>(c);
        if (id > kVocabsize) {
            tokens.push_back(GetUNKId());
        } else {
            tokens.push_back(id);
        }
    }
    return tokens;
}

std::string AsciiTokenizer::DecodeToken(int token_id) const {
    if (token_id == GetBOSId()) {
        return "<bos>";
    }
    if (token_id == GetEOSId()) {
        return "<eos>";
    }
    if (token_id == GetUNKId()) {
        return "<unk>";
    }
    // 控制用的ASCII码
    if (token_id < 32 || token_id == 127) {
        return "";
    }
    return std::string(1, static_cast<char>(token_id));
}

std::string AsciiTokenizer::Decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int token_id : token_ids) {
        result += DecodeToken(token_id);
    }
    return result;
}





}