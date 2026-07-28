#include "include/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <limits>

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

std::string GetUnicodePointFromUtf8(const std::string& s, size_t& pos) {
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

void SkipSpace(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<char>(s[pos]))) {
        pos++;
    }
}

bool GetJsonInt(const std::string& s, size_t& pos, int& out) {
    SkipSpace(s, pos);
    if (pos >= s.size()) {
        return false;
    }

    size_t start = pos;
    if (s[pos] == '-') {
        pos++;
    }
    while (pos < s.size() && std::isdigit(static_cast<char>(s[pos]))) {
        pos++;
    }
    if (pos == start) {
        return false;
    }
    try {
        out = std::stoi(s.substr(start, pos - start));
        return true;
    } catch (...) {
        return false;
    }
}


bool GetJsonString(const std::string& s, size_t& pos, std::string& out) {
    SkipSpace(s, pos);
    if (pos >= s.size() || s[pos] != '"') {
        return false;
    }
    ++pos;

    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            char next = s[pos + 1];
            switch (next) {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    if (pos + 5 < s.size()) {
                        std::string hex = s.substr(pos + 2, 4);
                        try {
                            char32_t unicode_point = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
                            out += UnicodepointToUtf8(unicode_point);
                        } catch(...) {
                            out += '?';
                        }
                        pos += 4;
                    }
                    break;
                }
                default:
                    out += next;
                    break;
            }
            pos += 2;
        } else {
            out += s[pos];
            pos++;
        }
    }

    if (pos < s.size() && s[pos] == '"') {
        pos++;
    }
    return true;
}
}  //namespace

void BpeTokenizer::BuildByteMap() {
    byte_to_unicode_ = BuildUtf8ByteToUnicodePoint();
    unicode_to_byte_.clear();
    for (int i = 0; i < 256; i++) {
        unicode_to_byte_[byte_to_unicode_[i]] = static_cast<u_int8_t>(i);
    }
}

void BpeTokenizer::LoadVocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        throw std::runtime_error("file is not open");
    }
    std::stringstream buf;
    buf << file.rdbuf();
    std::string content = buf.str();

    size_t pos = 0;
    SkipSpace(content, pos);
    if (pos >= content.size() || content[pos] != '{') {
        throw std::runtime_error("not find start in vocab_path");
    }
    //找到左边第一个大括号，跳过
    pos++;

    while (pos < content.size() && content[pos] != '}') {
        SkipSpace(content, pos);
        if (pos >= content.size() || content[pos] == '}') {
            break;
        }
        
        //得到key
        std::string key;
        if (!GetJsonString(content, pos, key)) {
            break;
        }

        SkipSpace(content, pos);
        if (pos >= content.size() || content[pos] != ':') {
            break;
        }
        // 找到key后面的冒号，跳过，后面就是value
        pos++;

        int val = 0;
        if (!GetJsonInt(content, pos, val)) {
            break;
        }
        vocab_[key] = val;

        SkipSpace(content, pos);
        //找完一条key-value后，将pos移到下一个的开始
        if (pos < content.size() && content[pos] == ',') {
            pos++;
        }
    }
    if (vocab_.empty()) {
        throw std::runtime_error("vocab is empty");
    }

    //为id和token的转换数组赋值
    int max_id = -1;
    for (auto& kv : vocab_) {
        if (kv.second > max_id) {
            max_id = kv.second;
        }
    }
    id_to_token_.resize(max_id + 1);
    for (auto& kv : vocab_) {
        id_to_token_[kv.second] = kv.first;
    }

    //将special token 赋值
    for (auto& kv : vocab_) {
        std::string s = kv.first;
        if (s.size() >= 4 && s.front() == '<' && s[1] == '|' 
            && s[s.size() - 2] == '|' && s[s.size() - 1] == '>') {
            special_tokens_.push_back({kv.first, kv.second});
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const auto& first, const auto& second) {
        return first.first.size() > second.first.size();
    });
}

void BpeTokenizer::LoadMergeRank(const std::string& merge_path) {
    std::ifstream file(merge_path);
    if (!file.is_open()) {
        throw std::runtime_error("merge file is not open");
    }
    std::string line;
    int rank = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string a, b;
        if (ss >> a >> b) {
            merge_ranks_[{a, b}] = rank++;
        }
    }
}

void BpeTokenizer::LoadSpecialTokensID(const std::string& special_path) {
    std::ifstream file(special_path);
    if (!file.is_open()) {
        throw std::runtime_error("special file is not open");
    }

    std::stringstream buf;
    buf << file.rdbuf();
    std::string content = buf.str();

    auto find_id = [&](const std::string key) {
        size_t pos = content.find('"' + key + '"');
        if (pos == std::string::npos) {
            return -1;
        }
        pos = content.find(':', pos);
        if (pos == std::string::npos) {
            return -1;
        }
        pos++;
        size_t start = pos;
        SkipSpace(content, pos);
        while (pos < content.size() && (std::isdigit(static_cast<char>(content[pos])) || content[pos] == '-')) {
            pos++;
        }
        size_t end = pos;
        if (start == end) {
            return -1;
        }
        return std::stoi(content.substr(start, end - start));
    };

    bos_id_ = find_id("bos_id");
    eos_id_ = find_id("eos_id");
    unk_id_ = find_id("unk_id");
}

bool BpeTokenizer::Load(const std::string& vocab_path, const std::string& merge_path, const std::string& special_path) {
    BuildByteMap();
    try {
        LoadVocab(vocab_path);
        LoadMergeRank(merge_path);
        LoadSpecialTokensID(special_path);        
    } catch (...) {
        return false;
    }

    std::cout << "BPE tokenizer load: vocab_size = " << vocab_.size()
              << " merges_size = " << merge_ranks_.size() << " bos_id = " << bos_id_
              << " eos_id = " << eos_id_ << " unk_id = " << unk_id_ << std::endl; 

    return true;
}

std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    std::vector<int> result;
    size_t pos = 0;
    while (pos < text.size()) {
        //查看是否为特殊符号
        bool is_match = false;
        for (auto& kv : special_tokens_) {
            std::string s = kv.first;
            if (pos + s.size() <= text.size() && 
                std::memcmp(text.data() + pos, s.data(), s.size()) == 0) {
                result.push_back(kv.second);
                pos += s.size();
                is_match = true;
                break;
            }
        }
        if (is_match) {
            continue;
        }

        //找到和开始特殊符号匹配的特殊符号，或者就到句子结束
        size_t end = text.size();
        for (auto& kv : special_tokens_) {
            size_t p = text.find(kv.first, pos);
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

        //BPE 算法
        while (words.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_index = words.size();
            for (int i = 0; i + 1 < words.size(); i++) {
                auto it = merge_ranks_.find({words[i], words[i + 1]});
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_index = i;
                }
            }
            if (best_index >= words.size()) {
                break;
            }

            words[best_index] += words[best_index + 1];
            words.erase(words.begin() + best_index + 1);
        }

        for (auto& word : words) {
            auto it = vocab_.find(word);
            if (it != vocab_.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(unk_id_);
            }
        }
        pos = end;  // advance past the processed segment
    }

    return result;
}


std::string BpeTokenizer::DecodeToken(int token_id) const {
    if (token_id < 0 || token_id >= static_cast<int>(id_to_token_.size())) {
        return "<unk>";
    }
    return id_to_token_[token_id];
}


std::string BpeTokenizer::Decode(const std::vector<int>& token_ids) const {
    std::string text;
    for (auto& token_id : token_ids) {
        text += DecodeToken(token_id);
    }

    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        std::string unicode_point = GetUnicodePointFromUtf8(text, pos);
        if (unicode_point.empty()) {
            continue;
        }
        auto it = unicode_to_byte_.find(unicode_point);
        if (it != unicode_to_byte_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += unicode_point;
        }
    }
    return result;
}

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path, const std::string& merges_path,const std::string& special_path) {
  auto tok = std::make_unique<BpeTokenizer>();
  if (tok->Load(vocab_path, merges_path, special_path)) {
    return tok;
  }
  return nullptr;
}

}