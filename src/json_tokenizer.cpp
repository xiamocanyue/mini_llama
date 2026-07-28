#include "include/tokenizer.h"

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace mini_llama {
namespace {

static std::string Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

static std::string GetJsonString(const std::string& content, size_t& pos) {
    while (pos < content.size() && content[pos] != '"') {
        pos++;
    }
    if (pos >= content.size() || content[pos] != '"') {
        throw std::runtime_error("content is not have string");
    }
    pos++; //跳过左边的双引号
    std::string result;
    while (pos < content.size() && content[pos] != '"') {
        //处理转义字符
        if (content[pos] == '\\' && pos +1 < content.size()) {
            char next = content[pos + 1];
            if (next == 'n') {
                result += '\n';
            } else if (next == 't') {
                result += "\t";
            } else if (next == 'r') {
                result += "\r";
            } else if (next == '\\') {
                result += "\\";
            } else if (next == '"') {
                result += '"';
            } else if (next == 'u' && pos + 5 < content.size()) { // 处理unicode编码
                std::string hex = content.substr(pos + 2, 4);
                unsigned int codepoint = std::stoul(hex, nullptr, 16);
                if (codepoint <= 0x7f) {
                    result += static_cast<char>(codepoint);
                } else {
                    result += '?';
                }
                pos += 6;
                continue;
            } else {
                result += next;
            }
            pos += 2;
        } else {
            result += content[pos];
            pos++;
        }
    }
    if (pos >= content.size()) {
        throw std::runtime_error("unterminated string");
    }
    //跳过右边的双引号
    pos++;
    return result;
}

static int GetJsonInt(const std::string& content, size_t& pos) {
    //跳过非数字部分
    while (pos < content.size() && !std::isdigit(static_cast<char>(content[pos])) && content[pos] != '-') {
        pos++;
    }
    size_t start = pos;
    if (content[pos] == '-') {
        pos++;
    }
    while (pos < content.size() && std::isdigit(static_cast<char>(content[pos]))) {
        pos++;
    }
    return std::stoi(content.substr(start, pos - start));
}

static bool GetJsonBool(const std::string& content, size_t& pos) {
    if (content.substr(pos, 4) == "true") {
        pos += 4;
        return true;
    } 
    if (content.substr(pos, 5) == "false") {
        pos += 5;
        return false;
    }
    throw std::runtime_error("it is not true or false");
}
}

JsonTokenizer::JsonTokenizer(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        throw std::runtime_error("fail to open vocab" + vocab_path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    if (file.bad()) {
        throw std::runtime_error("faile to read vocab");
    }

    std::string content = buffer.str();
    size_t pos = content.find('[');
    if (pos == std::string::npos) {
        throw std::runtime_error("vocab is not a array");
    }
    pos++;

    int max_id = -1;
    while (pos < content.size()) {
        //找到左边的大括号
        while (pos < content.size() && content[pos] != '{') {
            if (content[pos] == ']') {
                break;
            }
            pos++;
        }
        if (pos >= content.size() || content[pos] != '{') {
            break;
        }
        pos++;

        VocabEntry entry;
        bool has_id = false;
        bool has_content = false;
        while (pos < content.size() && content[pos] != '}') {
            //跳过空格和逗号
            while (pos < content.size() && (std::isspace(static_cast<char>(content[pos])) || content[pos] == ',')) {
                pos++;
            }
            if (pos >= content.size() || content[pos] == '}') {
                break;
            }
            //得到string的key
            std::string key = GetJsonString(content, pos);
            //找到冒号,并略过
            while (pos < content.size() && content[pos] != ':') {
                pos++;
            }
            if (pos < content.size()) {
                pos++;
            }
            //跳过冒号后的空格
            while (pos < content.size() && std::isspace(static_cast<char>(content[pos]))) {
                pos++;
            }
            //得到key后的value
            if (key == "id") {
                entry.id = GetJsonInt(content, pos);
                has_id = true;
            } else if (key == "content") {
                entry.content = GetJsonString(content, pos);
                has_content = true;
            } else if (key == "special") {
                entry.special = GetJsonBool(content, pos);
            } else {  //遇到不认识的
                if (pos < content.size() && content[pos] == '"') {
                    GetJsonString(content, pos);
                } else {
                    while (pos < content.size() && content[pos] != ',' && content[pos] != '}') {
                        pos++;
                    }                    
                }
            }
        }
        //处理完一行，跳过最右边的大括号
        if (pos < content.size() && content[pos] == '}') {
            pos++;
        }

        if (!has_content || !has_id) {
            throw std::runtime_error("this vocab is not have id or content");
        }
        if (entry.id > max_id) {
            max_id = entry.id;
        }
        if (entry.id < 0) {
            throw std::runtime_error("entry id is must be positive");
        }
        //为特殊符号编号赋值
        if (entry.content == "<bos>") {
            bos_id_ = entry.id;
        } else if (entry.content == "<eos>") {
            eos_id_ = entry.id;
        } else if (entry.content == "<unk>") {
            unk_id_ = entry.id;
        }
        //将一行的记录保存
        if (static_cast<size_t>(entry.id) >= id_to_entry_.size()) {
            id_to_entry_.resize(entry.id + 1);
        }
        id_to_entry_[entry.id] = entry;
        content_to_id_.push_back({entry.content, entry.id});
    }

    if (max_id < 0) {
        throw std::runtime_error("vocab file contains no data");
    }
    vocab_size_ = max_id + 1;
    std::sort(content_to_id_.begin(), content_to_id_.end(), [](const std::pair<std::string, int> & a, const std::pair<std::string, int> & b){
        return a.first.size() > b.first.size();
    });
}

std::vector<int> JsonTokenizer::Encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(GetBOSId());

    //最长字符匹配
    size_t pos = 0;
    while (pos < text.size()) {
        bool is_match = false;
        for (auto& pair : content_to_id_) {
            const std::string& content = pair.first;
            if (content.empty()) {
                continue;
            }
            if (text.compare(pos, content.size(), content) == 0) {
                tokens.push_back(pair.second);
                pos += content.size();
                is_match = true;
                break;
            }
        }     
        if (!is_match) {
            tokens.push_back(GetUNKId());
            pos++;
        }   
    }
    return tokens;
}

std::string JsonTokenizer::DecodeToken(int token_id) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= id_to_entry_.size()) {
        return "<unk>";
    }
    return id_to_entry_[token_id].content;
}

std::string JsonTokenizer::Decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int token_id : token_ids) {
        result += DecodeToken(token_id);
    }
    return result;
}


std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path) {
  if (!vocab_path.empty()) {
    std::ifstream f(vocab_path);
    if (f.good()) {
      try {
        return std::make_unique<JsonTokenizer>(vocab_path);
      } catch (const std::exception& e) {
        std::cerr << "Warning: failed to load vocab '" << vocab_path
                  << "', falling back to ASCII tokenizer: " << e.what()
                  << std::endl;
      }
    }
  }
  return std::make_unique<AsciiTokenizer>();
}

}

