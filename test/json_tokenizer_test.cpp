#include "include/tokenizer.h"

#include <string>
#include <iostream>

using namespace mini_llama;

int main() {
    JsonTokenizer jt("models/tiny/vocab.json");
    std::string prompt = "hello world!";
    std::vector<int> result;
    result = jt.Encode(prompt);
    std::cout << "encode result :  " ;
    for (int n : result) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    std::cout << "decode result : " << jt.Decode(result) << std::endl;

    return 0;
}