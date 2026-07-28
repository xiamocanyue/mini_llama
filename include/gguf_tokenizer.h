#pragma once 

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>

#include "include/tokenizer.h"

namespace mini_llama {

class GgufTokenizer : public ITokenizer {
public: 
     GgufTokenizer() = default;
     ~GgufTokenizer() = default;
     bool Load(const std::string& gguf_path);

     std::vector<int> Encode(const std::string& text) const override;
     std::string DecodeToken(int token_id) const override;
     std::string Decode(const std::vector<int>& token_ids) const override;

     int GetVocabSize() const override {
          return static_cast<int>(id_to_token_.size());
     }
     int GetBOSId() const override {
          return bos_id_;
     }
     int GetEOSId() const override {
          return eos_id_;
     }
     int GetUNKId() const override {
          return unk_id_;
     }

private:
     void BuildByteMap();
    


private:
     std::unordered_map<std::string, int> vocab_;
     std::map<std::pair<std::string, std::string>, int> merge_ranks_;
     std::vector<std::string> id_to_token_;
     std::vector<std::pair<std::string, int>> special_tokens_;

     int bos_id_ = -1;
     int eos_id_ = -1;
     int unk_id_ = -1;

     std::vector<std::string> byte_to_unicode_;
     std::unordered_map<std::string, u_int8_t> unicode_to_byte_;
};

std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path);

}