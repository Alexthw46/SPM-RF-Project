#pragma once
#include <vector>
#include <random>
#include "Node.hpp"

class DecisionTree {
public:
    explicit DecisionTree(int max_depth_ = 5, int min_samples_ = 2,
                          unsigned int seed = std::random_device{}());
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    [[nodiscard]] int predict(const std::vector<double>& x) const;

    [[nodiscard]] std::vector<int> predict_batch(const std::vector<std::vector<double>> &X) const;

private:
    int max_depth;
    int min_samples_split;
    Node* root;
    std::mt19937 gen;

    Node* build(const std::vector<std::vector<double>>& X,
                const std::vector<int>& y, int depth);
    static double gini(const std::vector<int>& y);
    static double split_score(const std::vector<std::vector<double>>& X,
                              const std::vector<int>& y, int f, double t);
    static int majority_label(const std::vector<int>& y);

    static int predict_one(const Node* node, const std::vector<double>& x);

};
