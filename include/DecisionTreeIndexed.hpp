#pragma once
#include <vector>
#include <random>
#include <unordered_map>
#include "Node.hpp"


/**
 * @brief Lightweight view wrapper for column-major data.
 *
 * Provides read-only access to a 2D dataset stored in column-major layout.
 * Used to access a transposed row-major dataset without changing the original access patterns.
 * (outer vector indexes features, inner vector indexes samples).
 */
struct ColMajorView {
    /** Reference to column-major data: outer index = feature, inner index = sample. */
    const std::vector<std::vector<double> > &data;

    /**
     * @brief Access the value for sample `i` and feature `f`.
     * @param i Sample index (row in a logical row-major view).
     * @param f Feature index (column).
     * @return The value at data[f][i].
     */
    double operator()(const size_t i, const size_t f) const { return data[f][i]; }
};

struct RowMajorView {
    const std::vector<std::vector<double> > &data;

    double operator()(const size_t i, const size_t f) const { return data[i][f]; }
};

class DecisionTree {
public:
    DecisionTree(int max_depth_, int min_samples_, unsigned int seed);

    template<class View>
    void fit(const View &Xc, const std::vector<int> &y,
             const std::vector<size_t> &indices); // build from indices
    [[nodiscard]] int predict(const std::vector<double> &x) const;

    Node *root = nullptr;

private:
    int max_depth;
    int min_samples;
    std::mt19937 gen;

    [[nodiscard]] static double gini_from_counts(const std::unordered_map<int, int> &counts, size_t n_features);

    [[nodiscard]] static double gini(const std::vector<int> &y);

    [[nodiscard]] static int majority_label_from_counts(const std::unordered_map<int, int> &counts);

    template<class View>
    Node *build(const View &Xc,
                const std::vector<int> &y,
                const std::vector<size_t> &indices,
                int depth);

    static int predict_one(const Node *node, const std::vector<double> &x);
};
