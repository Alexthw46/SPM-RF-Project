#pragma once

struct Node {
    bool is_leaf{false};
    int feature{-1};
    double threshold{0};
    int label{-1};
    Node* left{nullptr};
    Node* right{nullptr};

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
    int right{ -1 }; // index of right child in flat array
};
