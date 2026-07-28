#include "include/batch.h"


namespace mini_llama {
    MiniBatch MiniBatch::GetSingleMiniBatch(int token, int pos) {
        MiniBatch batch;
        batch.tokens.push_back(token);
        batch.positions.push_back(pos);
        return batch;
    }

    MiniBatch MiniBatch::GetMultiMiniBatch(const std::vector<int>& tokens, int start_pos) {
        MiniBatch batch;
        batch.tokens = tokens;
        batch.positions.reserve(tokens.size());
        for (int i = 0; i < tokens.size(); i++) {
            batch.positions.push_back(start_pos + i);
        }
        return batch;
    }


    
}