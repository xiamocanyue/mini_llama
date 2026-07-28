#pragma once 

#include <vector>

namespace mini_llama {

struct MiniBatch {
    static MiniBatch GetSingleMiniBatch(int token, int pos);
    static MiniBatch GetMultiMiniBatch(const std::vector<int>& tokens, int start_pos = 0);

    int GetTokensNum() const {
        return static_cast<int>(tokens.size());
    }

    std::vector<int> tokens;
    std::vector<int> positions;
};    

}