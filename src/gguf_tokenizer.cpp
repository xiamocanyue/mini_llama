#include "include/gguf_tokenizer.h"

#include <algorithm>
#include <iostream>
#include <cstring>
#include <limits>

#include "include/gguf.h"


namespace mini_llama {
namespace {

//7位unicode码点   ->   0xxxxxxx
//11位             ->  110xxxxx    10xxxxxx
//16位             ->  1110xxxx    10xxxxxx  10xxxxxx
//21位             ->  11110xxx    10xxxxxx  10xxxxxx  10xxxxxx
//实际上这里最多用到第二分支，即cp <= 256+256 = 512
std::string UnicodepointToUtf8(char32_t cp) {
    std::string result;
    if (cp <= 0x7f) {
        result += static_cast<char>(cp);
    } else if (cp <= 0x7ff) {
        result += static_cast<char>(0xc0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        result += static_cast<char>(0xe0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        result += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
        result += static_cast<char>(0xf0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3f));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        result += static_cast<char>(0x80 | (cp & 0x3f));
    }
    return result;
}

std::string GetUnicodeStringFromUtf8(const std::string& s, size_t& pos) {
    if (pos >= s.size()) {
        return "";
    }
    unsigned char first = s[pos];
    size_t len = 1;
    //看前缀是不是 0, 110, 1110, 11110
    if ((first & 0x80) == 0) {
        len = 1;
    } else if ((first & 0xe0) == 0xc0) {
        len = 2;
    } else if ((first & 0xf0) == 0xe0) {
        len = 3;
    } else if ((first & 0xf8) == 0xf0) {
        len = 4;
    } else {
        pos++;
        return "";
    }
    std::string result = s.substr(pos, len);
    pos += len;
    return result;
}

std::vector<std::string> BuildUtf8ByteToUnicodePoint() {
    std::vector<std::string> map(256);
    std::vector<int> utf8_safe_bytes;
    //安全的utf8
    for (int c = '!'; c <= '~'; c++) {
        utf8_safe_bytes.push_back(c);
    }
    for (int c = 0xa1; c <= 0xac; c++) {
        utf8_safe_bytes.push_back(c);
    }
    for (int c = 0xae; c <= 0xff; c++) {
        utf8_safe_bytes.push_back(c);
    }

    std::vector<int> unicode_points = utf8_safe_bytes;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        //不安全的utf8，比如0 -> 32 回车，空格，换行
        //+ 256 变成安全的可打印的字符
        if (std::find(utf8_safe_bytes.begin(), utf8_safe_bytes.end(), b) == utf8_safe_bytes.end()) {
            utf8_safe_bytes.push_back(b);
            unicode_points.push_back(256 + n);
            n++;
        }
    }

    for (int i = 0; i < utf8_safe_bytes.size(); i++) {
        map[utf8_safe_bytes[i]] = UnicodepointToUtf8(static_cast<char32_t>(unicode_points[i]));
    }
    return map;
}

} // namespace 

void GgufTokenizer::BuildByteMap() {
    byte_to_unicode_ = BuildUtf8ByteToUnicodePoint();
    unicode_to_byte_.clear();
    for (int i = 0; i < 256; i++) {
        unicode_to_byte_[byte_to_unicode_[i]] = static_cast<uint8_t>(i);
    }
}

bool GgufTokenizer::Load(const std::string& gguf_path) {
    vocab_.clear();
    unicode_to_byte_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();
    bos_id_ = -1;
    eos_id_ = -1;
    unk_id_ = -1;
    //加载 unicode_to_byte_ 和 byte_to_unicode_
    BuildByteMap();


    //加载id_to_tokens 和 vocab_ 
    std::vector<std::string> tokens;
    if (!GetGgufMetaDataStringArray(gguf_path, "tokenizer.ggml.tokens", tokens) || tokens.empty()) {
        std::cout << "GGUF tokenizer metadata missing token list; falling back to tiny vocab." << std::endl;
        return false;
    }
    id_to_token_.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        vocab_[tokens[i]] = static_cast<int>(i);
        id_to_token_[i] = tokens[i];
    }

    //加载special_tokens  
    std::vector<int> token_types;
    bool has_token_types = GetGgufMetaDataIntArray(gguf_path, "tokenizer.ggml.token_type", token_types);
    if (has_token_types && token_types.size() == tokens.size()) {
        for (size_t i = 0; i < tokens.size(); i++) {
            if (token_types[i] == 3) {
                special_tokens_.push_back({tokens[i], static_cast<int>(i)});
            }
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const auto& a, const auto& b) {
        return a.first.size() > b.first.size();
    });

    //加载merge_ranks_
    std::vector<std::string> merges;
    if (GetGgufMetaDataStringArray(gguf_path, "tokenizer.ggml.merges", merges)) {
        for (int rank = 0; rank < static_cast<int>(merges.size()); rank++) {
            std::string& s = merges[rank];
            size_t space_pos = s.find(' ');
            if (space_pos == std::string::npos) {
                continue;
            }
            std::string left = s.substr(0, space_pos);
            std::string right = s.substr(space_pos + 1);
            merge_ranks_[{left, right}] = rank;
        }
    }

    //加载 特殊id
    int64_t bos = -1, eos = -1, unk = -1;
    if (GetGgufMetaDataInt(gguf_path, "tokenizer.ggml.bos_token_id", bos)) {
        bos_id_ = static_cast<int>(bos);
    }
    if (GetGgufMetaDataInt(gguf_path, "tokenizer.ggml.eos_token_id", eos)) {
        eos_id_ = static_cast<int>(eos);
    }
    if (GetGgufMetaDataInt(gguf_path, "tokenizer.ggml.unknown_token_id", unk)) {
        unk_id_ = static_cast<int>(unk);
    }

  std::cout << "GGUF tokenizer loaded: vocab=" << vocab_.size()
            << " merges=" << merge_ranks_.size()
            << " special=" << special_tokens_.size() << " bos=" << bos_id_
            << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
  return true;
}

std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    std::vector<int> result;
    size_t pos = 0;
    while (pos < text.size()) {

        //先匹配特殊字符
        bool is_match = false;
        for (auto& [special_token_str, token_id] : special_tokens_) {
            if (pos + special_token_str.size() <= text.size() && std::memcmp(text.data() + pos, special_token_str.data(), special_token_str.size() == 0)) {
                result.push_back(token_id);
                pos += special_token_str.size();
                is_match = true;
                break;
            } 
        }
        if (is_match) {
            continue;
        }

        //找到和开始特殊符号匹配的特殊符号，或者就到句子结束
        size_t end = text.size();
        for (auto& [special_str, id] : special_tokens_) {
            size_t p = text.find(special_str, pos);
            if (p != std::string::npos && p < end) {
                end = p;
            }
        }

        std::string segment = text.substr(pos, end - pos);

        std::vector<std::string> words;
        for (unsigned char c : segment) {
            if (c >= byte_to_unicode_.size()) {
                throw std::runtime_error("byte is over byte_to_unicode_ size 256");
                return {};
            }
            words.push_back(byte_to_unicode_[c]);
        }

        //BPE merge
        while (words.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t merge_index = words.size();
            for (size_t i = 0; i + 1 < words.size(); i++) {
                auto it = merge_ranks_.find({words[i], words[i + 1]});
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    merge_index = i;
                }
            }
            if (merge_index >= words.size()) {
                break;
            }
            words[merge_index] += words[merge_index + 1];
            words.erase(words.begin() + merge_index + 1);
        }

        for (auto& word : words) {
            auto it = vocab_.find(word);
            if (it != vocab_.end()) {
                result.push_back(it->second);
            } else  {
                result.push_back(unk_id_);
            }
        }
    }
    return result;
}

std::string GgufTokenizer::DecodeToken(int token_id) const {
    if (token_id >= 0 && token_id < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[token_id];
    }
    return "";
}

std::string GgufTokenizer::Decode(const std::vector<int>& token_ids) const {
    std::string text;
    for (auto id : token_ids) {
        text += DecodeToken(id);
    }

    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = pos;
        std::string unicode_str = GetUnicodeStringFromUtf8(text, pos);
        if (unicode_str.empty()) {
            if (pos == start) {
                pos++;
            }
            continue;
        }
        auto it = unicode_to_byte_.find(unicode_str);
        if (it != unicode_to_byte_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += unicode_str;
        }
    }
    return result;
}


std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path) {
    auto tok = std::make_unique<GgufTokenizer>();
    if (tok->Load(gguf_path)) {
        return tok;
    }
    return nullptr;
}

}