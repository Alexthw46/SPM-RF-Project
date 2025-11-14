#pragma once
#include <vector>
#include <random>
#include <unordered_map>
#include "Node.hpp"

class DecisionTree {
public:
    DecisionTree(int max_depth_, int min_samples_, unsigned int seed);

    void fit(const std::vector<std::vector<double> > &X, const std::vector<int> &y,
             const std::vector<size_t> &indices); // build from indices
    [[nodiscard]] int predict(const std::vector<double> &x) const;

    Node *root = nullptr;

private:
    int max_depth;
    int min_samples;
    std::mt19937 gen;

    [[nodiscard]] static double gini_from_counts(const std::unordered_map<int, int> &counts, size_t total);

    [[nodiscard]] static double gini(const std::vector<int> &y);

    [[nodiscard]] static int majority_label_from_counts(const std::unordered_map<int, int> &counts);

    Node *build(const std::vector<std::vector<double> > &X,
                const std::vector<int> &y,
                const std::vector<size_t> &indices,
                int depth);

    static int predict_one(const Node *node, const std::vector<double> &x);
};
