#pragma once
#include <vector>
#include <random>
#include "DecisionTree.hpp"

class RandomForest {
public:
    explicit RandomForest(int n_t = 5, int max_depth = 5, int n_classes = 2,
                          unsigned int seed = std::random_device{}());
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    [[nodiscard]] int predict(const std::vector<double>& x) const;

    [[nodiscard]] std::vector<int> predict_batch(const std::vector<std::vector<double>> &X) const;

private:
    int n_trees;
    int max_depth;
    int n_classes;
    std::vector<DecisionTree> trees;
    std::mt19937 gen;
};
