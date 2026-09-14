#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/tokenizer.h"

using namespace mini_llama;

int main() {
    try {
        JsonTokenizer tokenizer("models/tiny/vocab.json");
        const std::vector<int> expected = {1, 104, 101, 108, 108, 111, 33};
        const std::vector<int> actual = tokenizer.Encode("hello!");
        if (actual != expected) {
            throw std::runtime_error("encoded token sequence mismatch");
        }
        if (tokenizer.Decode(actual) != "<bos>hello!") {
            throw std::runtime_error("decoded text mismatch");
        }
        if (tokenizer.DecodeToken(104) != "h" ||
            tokenizer.DecodeToken(tokenizer.GetUNKId()) != "<unk>") {
            throw std::runtime_error("single-token decode mismatch");
        }
        if (tokenizer.GetVocabSize() != 128 || tokenizer.GetBOSId() != 1 ||
            tokenizer.GetEOSId() != 2 || tokenizer.GetUNKId() != 0) {
            throw std::runtime_error("tokenizer metadata mismatch");
        }
        std::cout << "json_tokenizer_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "json_tokenizer_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
