#pragma once 

#include <vector>
#include <map>
#include <memory>

namespace mini_llama {

class RadixTree {
public:
    struct Node {
        std::vector<int> key;
        bool terminal = false;
        std::map<int, std::unique_ptr<Node>> children;
    };

    RadixTree() = default;
    ~RadixTree() = default;

    RadixTree(const RadixTree&) = delete;
    RadixTree& operator=(const RadixTree&) = delete;    

    RadixTree(RadixTree&&) noexcept;
    RadixTree& operator=(RadixTree&&) noexcept;

    void Clear();
    bool Empty();
    size_t Size();
    void Insert(const std::vector<int>& tokens);
    size_t GetLongestPrefix(const std::vector<int>& tokens);
    static size_t GetCommonPrefixLength(const std::vector<int>& a, size_t a_offset, const std::vector<int>& b);

private:
    void InsertInto(Node* parent, const std::vector<int>& tokens);
    


private:
    Node root_;
    size_t terminal_cout_ = 0;
};

}