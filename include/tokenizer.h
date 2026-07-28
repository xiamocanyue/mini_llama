#pragma once 

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>

namespace mini_llama {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int> Encode(const std::string& text) const = 0;
    virtual std::string DecodeToken(int token_id) const = 0;
    virtual std::string Decode(const std::vector<int>& token_ids) const = 0;

    virtual int GetVocabSize() const = 0;
    virtual int GetBOSId() const = 0;
    virtual int GetEOSId() const = 0;
    virtual int GetUNKId() const = 0;
};

class AsciiTokenizer : public ITokenizer {
public:
    static constexpr int kVocabsize = 128;
    AsciiTokenizer() = default;

    std::vector<int> Encode(const std::string& text) const override;
    std::string Decode(const std::vector<int>& token_ids) const override;
    std::string DecodeToken(int token_id) const override;
    int GetVocabSize() const override {
        return kVocabsize;
    }
    int GetBOSId() const override {
        return 1;
    }
    int GetEOSId() const override {
        return 2;
    }
    int GetUNKId() const override {
        return 0;
    }
};


/*
[
  {"id": 0, "content": "<unk>", "special": true},
  {"id": 1, "content": "<bos>", "special": true},
  {"id": 2, "content": "<eos>", "special": true},
  {"id": 3, "content": "hello", "special": false}
]
*/
class JsonTokenizer : public ITokenizer {
public:
    explicit JsonTokenizer(const std::string& vocab_path);

    std::vector<int> Encode(const std::string& text) const override;
    std::string Decode(const std::vector<int>& token_ids) const override;
    std::string DecodeToken(int token_id) const override;
    int GetVocabSize() const override {
        return vocab_size_;
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
    struct VocabEntry {
        int id = 0;
        std::string content;
        bool special = false;
    };
    
    int vocab_size_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;

    std::vector<VocabEntry> id_to_entry_;
    std::vector<std::pair<std::string, int>> content_to_id_;
};


class BpeTokenizer : public ITokenizer {
public:
    BpeTokenizer() = default;

    bool Load(const std::string& vocab_path, const std::string& merge_path, const std::string& special_path);
    std::vector<int> Encode(const std::string& text) const override;
    std::string Decode(const std::vector<int>& token_ids) const override;
    std::string DecodeToken(int token_id) const override;
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
    void LoadVocab(const std::string& vocab_path);
    void LoadMergeRank(const std::string& merge_path);
    void LoadSpecialTokensID(const std::string& special_path);


private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;
    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    std::vector<std::pair<std::string, int>> special_tokens_;
    std::vector<std::string> byte_to_unicode_;
    std::unordered_map<std::string, u_int8_t> unicode_to_byte_;

};

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path, const std::string& merges_path, const std::string& special_path);
   

}