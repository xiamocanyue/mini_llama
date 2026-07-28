#include "include/radix_tree.h"

namespace mini_llama {

RadixTree::RadixTree(RadixTree&&) noexcept = default;
RadixTree& RadixTree::operator=(RadixTree&&) noexcept = default;

void RadixTree::Clear() {
    root_.children.clear();
    root_.key.clear();
    root_.terminal = false;
    terminal_cout_ = 0;
}

bool RadixTree::Empty() {
    return terminal_cout_ == 0;
}

size_t RadixTree::Size() {
    return terminal_cout_;
}

size_t RadixTree::GetCommonPrefixLength(const std::vector<int>& a, size_t a_offset, const std::vector<int>& b) {
    size_t i = 0;
    while (a_offset + i < a.size() && i < b.size() && a[a_offset + i] == b[i]) {
        i++;
    }
    return i;
}

void RadixTree::Insert(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        if (root_.terminal == false) {
            root_.terminal = true;
            terminal_cout_++;
        }
        return ;
    }
    InsertInto(&root_, tokens);
}

void RadixTree::InsertInto(Node* parent, const std::vector<int>& tokens) {
    //要添加的和已经存在的没有重复token
    auto find_result = parent->children.find(tokens[0]);
    if (find_result == parent->children.end()) {
        auto new_child = std::make_unique<Node>();
        new_child->key = std::move(tokens);
        new_child->terminal = true;
        parent->children[tokens[0]] = std::move(new_child);
        terminal_cout_++;
        return ;
    }

    //有重复token
    std::unique_ptr<Node>& child = find_result->second;
    size_t common_length = GetCommonPrefixLength(tokens, 0, child->key);
    //插入的和公共前缀完全匹配
    if (common_length == child->key.size()) {
        //插入的和前缀一样长
        if (common_length == tokens.size()) {
            if (child->terminal == false) {
                child->terminal = true;
                terminal_cout_++;
                return ;
            }
        }
        //插入的比当前的前缀长
        std::vector<int> different_arr(tokens.begin() + common_length, tokens.end());
        InsertInto(child.get(), different_arr);
        return ;
    }
    /*插入的比公共前缀小，或者公共前缀没有完全匹配，要截断child
        child [1,2,3,4]  
        公共前缀为 [1,2]
        spilt {
            key = [1,2];
            child {
                3 -> [3,4],
                8 -> [8,9]
            }
            terminal = false;
        }
    */
    auto spilt = std::make_unique<Node>();
    spilt->key.assign(tokens.begin(), tokens.begin() + common_length);
    child->key.erase(child->key.begin(), child->key.begin() + common_length);
    spilt->children[child->key[0]] = std::move(child);
    // tokens [1,2] 
    if (common_length == tokens.size()) {
        spilt->terminal = true;
        terminal_cout_++;
    } else { //tokens [1,2,8,9]   different_arr = [8,9]
        std::vector<int> different_arr(tokens.begin() + common_length, tokens.end());
        InsertInto(spilt.get(), different_arr);  
    }
    find_result->second = std::move(spilt); 
    return ;     
}

size_t RadixTree::GetLongestPrefix(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        return 0;
    }

    Node* node = &root_;
    size_t pos = 0;
    while (pos < tokens.size()) {
        auto find_result = node->children.find(tokens[pos]);
        if (find_result == node->children.end()) {
            return pos;
        }

        Node* child = find_result->second.get();
        size_t common_length = GetCommonPrefixLength(tokens, pos, child->key);
        pos += common_length;
        if (common_length < child->key.size()) {
            return pos;
        }
        node = child;
    }
    return pos;
}

}