#pragma once
#include <functional>

struct Node {
    bool is_leaf{false};
    int feature{-1};
    double threshold{0};
    int label{-1};
    Node *left{nullptr};
    Node *right{nullptr};

    // Destructor: recursively delete children
    ~Node() {
        delete left;
        delete right;
    }
};

struct FlatNode {
    bool is_leaf{false};
    int feature{-1};
    double threshold{0};
    int label{-1};
    int right{-1}; // index of right child in flat array

    // Hash function for FlatNode
    static size_t hashNode(const FlatNode &n) {
        size_t h = std::hash<bool>{}(n.is_leaf);
        h ^= std::hash<int>{}(n.feature) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<double>{}(n.threshold) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(n.label) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(n.right) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
