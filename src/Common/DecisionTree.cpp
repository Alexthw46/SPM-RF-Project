#include "Node.hpp"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include "DecisionTreeIndexed.hpp"

using namespace std;

DecisionTree::DecisionTree(const int max_depth_, const int min_samples_, const int min_samples_leaf_,
                           const int n_classes, const unsigned int seed)
    : max_depth(max_depth_), min_samples_split(min_samples_), min_samples_leaf(min_samples_leaf_), n_classes(n_classes),
      gen(seed) {
    n_features = n_try = 1; // real init is in fit
}

// Recursive tree build
// Constructs the decision tree recursively, one version for multiclass and one opt. for binary classification
// Splits data based on best gini impurity of a feature, selected randomly
// Then calls itself for left and right child nodes, until stopping criteria met
// Measures and logs time taken for building the tree
template<class View>
Node *DecisionTree::build(const View &Xc,
                          const std::vector<int> &y,
                          const std::vector<size_t> &indices,
                          const int depth) {
    // Build label counts for this node
    std::vector counts(n_classes, 0);
    for (const size_t idx: indices) counts[y[idx]]++;

    // Stopping criteria
    if (depth >= max_depth || indices.size() <= static_cast<size_t>(min_samples_split) || gini_from_counts(
            counts, indices.size()) == 0.0) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label_from_counts(counts);
        return leaf;
    }

    uniform_int_distribution feature_dist(0, n_features - 1);

    // Find best split
    int best_f = -1;
    double best_t = 0.0;
    double best_score = numeric_limits<double>::max();

    // Pre-allocate count vectors that will be reused
    std::vector<int> left_counts(n_classes);
    std::vector<int> right_counts(n_classes);

    const size_t val_size = indices.size();
    std::vector<pair<double, size_t> > vals(val_size);

    // Start from sqrt(n_features) attempts and increase if needed, copy to avoid conflict
    int n_try_node = n_try;
    // For each candidate feature chosen randomly
    for (int k = 0; k < n_try_node; ++k) {
        const int f = feature_dist(gen);

        // collect (value, index) pairs for this feature
        for (size_t i = 0; i < val_size; ++i) {
            size_t idx = indices[i];
            vals[i].first = Xc(idx, f);
            vals[i].second = idx;
        }

        // sort by feature value to skip identical values
        std::sort(vals.begin(), vals.end(), [](auto &a, auto &b) { return a.first < b.first; });

        // left_counts: counts of labels on the left side of the split
        // right_counts: counts of labels on the right side of the split
        // initialize right to total counts and shrinks as samples move to left
        ranges::fill(left_counts, 0); // reset
        right_counts = counts; // copy

        // Keep track of number of samples on left
        size_t left_n = 0;

        // iterate potential split positions (only unique values)
        for (size_t i = 1; i < val_size; ++i) {
            // move sample i-1 to left side
            const int label_i_1 = y[vals[i - 1].second];
            int *__restrict left = left_counts.data();
            int *__restrict right = right_counts.data();

            left[label_i_1] += 1;
            right[label_i_1] -= 1;
            ++left_n;

            const double curr = vals[i].first;
            const double prev = vals[i - 1].first;

            // skip identical values for threshold
            if (curr == prev) continue;

            // set threshold
            const double t = 0.5 * (curr + prev);

            // compute Gini efficiently using prefix counts,
            // inlined method version to optimize operations
            const double inv_left_n = 1.0 / static_cast<double>(left_n);
            const double inv_right_n = 1.0 / static_cast<double>(val_size - left_n);
            double sumL = 0.0;
            double sumR = 0.0;
            for (int c = 0; c < n_classes; ++c) {
                const double pL = static_cast<double>(left_counts[c]) * inv_left_n;
                sumL += pL * pL;
                const double pR = static_cast<double>(right_counts[c]) * inv_right_n;
                sumR += pR * pR;
            }
            const double gL = 1.0 - sumL;
            const double gR = 1.0 - sumR;
            // cast to double to avoid integer division
            const double inv_val_size = 1.0 / static_cast<double>(val_size);
            const double wL = static_cast<double>(left_n) * inv_val_size;
            const double wR = 1.0 - wL;
            if (const double score = wL * gL + wR * gR; score < best_score) {
                best_score = score;
                best_f = f;
                best_t = t;
            }
        }

        // if we are in the last try and there is still not a valid split, keep scrolling the features until n_features
        if (k == n_try_node - 1 && best_f == -1 && n_try_node < n_features) {
            ++n_try_node; // increase the number of tries
        }
    }

    // If no valid split found, create leaf with majority label
    if (best_f == -1) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label_from_counts(counts);
        return leaf;
    }

    // Partition indices into left/right
    std::vector<size_t> left_idx;
    left_idx.reserve(indices.size() / 2);
    std::vector<size_t> right_idx;
    right_idx.reserve(indices.size() / 2);
    for (size_t idx: indices) {
        if (Xc(idx, best_f) <= best_t) left_idx.push_back(idx);
        else right_idx.push_back(idx);
    }

    // If either side is empty, create leaf with majority label
    if (left_idx.size() < static_cast<size_t>(min_samples_leaf) || right_idx.size() < static_cast<size_t>(
            min_samples_leaf)) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label_from_counts(counts);
        return leaf;
    }

    // Add decision node with the best feature/threshold
    const auto node = new Node();
    node->feature = best_f;
    node->threshold = best_t;

    // Recursively build left and right subtrees using the partitioned indices
    node->left = build(Xc, y, left_idx, depth + 1);
    node->right = build(Xc, y, right_idx, depth + 1);
    return node;
}


template<class View>
Node *DecisionTree::build_bin(const View &Xc,
                              const std::vector<int> &y,
                              const std::vector<size_t> &indices,
                              const int depth) {
    // Build label counts for this node
    int total_pos = 0;
    for (const size_t idx: indices) total_pos += y[idx];
    const size_t n = indices.size();

    // Stopping criteria
    if (depth >= max_depth ||
        n <= static_cast<size_t>(min_samples_split) ||
        total_pos == 0 || total_pos == static_cast<int>(n)) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = (total_pos * 2 >= static_cast<int>(n));
        return leaf;
    }

    uniform_int_distribution feature_dist(0, n_features - 1);

    // Find best split
    int best_f = -1;
    double best_t = 0.0;
    double best_score = numeric_limits<double>::max();

    const size_t val_size = n;
    std::vector<pair<double, size_t> > vals(val_size);

    // Start from sqrt(n_features) attempts and increase if needed, copy to avoid conflict
    int n_try_node = n_try;
    // For each candidate feature chosen randomly
    for (int k = 0; k < n_try_node; ++k) {
        const int f = feature_dist(gen);

        // collect (value, index) pairs for this feature
        for (size_t i = 0; i < val_size; ++i) {
            const size_t idx = indices[i];
            vals[i].first = Xc(idx, f);
            vals[i].second = idx;
        }

        // sort by feature value to skip identical values
        std::sort(vals.begin(), vals.end(),
                  [](auto &a, auto &b) { return a.first < b.first; });

        // initialize right to total counts and shrinks as samples move to left
        int left_pos = 0;
        int right_pos = total_pos;

        // Keep track of number of samples on left
        double left_n = 0;

        // iterate potential split positions (only unique values)
        for (size_t i = 1; i < val_size; ++i) {
            // move sample i-1 to left side
            const int yi = y[vals[i - 1].second];
            left_pos += yi;
            right_pos -= yi;
            ++left_n;

            const double curr = vals[i].first;
            const double prev = vals[i - 1].first;

            // skip identical values for threshold
            if (curr == prev) continue;

            const double right_n = static_cast<double>(val_size) - left_n;
            if (left_n == 0 || right_n == 0) continue;

            // compute Gini efficiently using prefix counts
            const double pL = static_cast<double>(left_pos) / left_n;
            const double pR = static_cast<double>(right_pos) / right_n;

            const double gL = 2.0 * pL * (1.0 - pL);
            const double gR = 2.0 * pR * (1.0 - pR);

            const double wL = left_n / static_cast<double>(val_size);
            const double wR = 1.0 - wL;

            if (const double score = wL * gL + wR * gR; score < best_score) {
                best_score = score;
                best_f = f;
                best_t = 0.5 * (curr + prev);
            }
        }

        // if we are in the last try and there is still not a valid split,
        // keep scrolling the features until n_features
        if (k == n_try_node - 1 && best_f == -1 && n_try_node < n_features) {
            ++n_try_node;
        }
    }

    // If no valid split found, create leaf with majority label
    if (best_f == -1) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = (total_pos * 2 >= static_cast<int>(n));
        return leaf;
    }

    // Partition indices into left/right
    std::vector<size_t> left_idx;
    left_idx.reserve(indices.size() / 2);
    std::vector<size_t> right_idx;
    right_idx.reserve(indices.size() / 2);
    for (size_t idx: indices) {
        if (Xc(idx, best_f) <= best_t) left_idx.push_back(idx);
        else right_idx.push_back(idx);
    }

    // If either side is empty, create leaf with majority label
    if (left_idx.size() < static_cast<size_t>(min_samples_leaf) ||
        right_idx.size() < static_cast<size_t>(min_samples_leaf)) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = (total_pos * 2 >= static_cast<int>(n));
        return leaf;
    }

    // Add decision node with the best feature/threshold
    const auto node = new Node();
    node->feature = best_f;
    node->threshold = best_t;

    // Recursively build left and right subtrees using the partitioned indices
    node->left = build_bin(Xc, y, left_idx, depth + 1);
    node->right = build_bin(Xc, y, right_idx, depth + 1);
    return node;
}

int DecisionTree::predict_one(const Node *node, const vector<double> &x) {
    if (!node) {
        std::cerr << "ERROR: predict_one called with null node!" << std::endl;
        return -1;
    }
    // Base case: leaf node -> return label
    if (node->is_leaf) return node->label;
    // Recursive case: traverse left or right child based on feature threshold
    return predict_one(x[node->feature] <= node->threshold ? node->left : node->right, x);
}

int DecisionTree::predict_flat(const std::vector<double> &x) const {
    int idx = 0;
    for (;;) {
        const auto &[threshold, feature, label, right, is_leaf] = flat[idx];
        if (is_leaf)
            return label;

        idx = x[feature] <= threshold
                  ? idx + 1 // left child is always next
                  : right; // precomputed right child index
    }
}

template Node *DecisionTree::build<ColMajorViewFlat>(
    const ColMajorViewFlat &,
    const std::vector<int> &,
    const std::vector<size_t> &,
    int);

template Node *DecisionTree::build_bin<ColMajorViewFlat>(
    const ColMajorViewFlat &,
    const std::vector<int> &,
    const std::vector<size_t> &,
    int);

template void DecisionTree::fit<ColMajorViewFlat>(
    const ColMajorViewFlat &,
    const std::vector<int> &,
    const std::vector<size_t> &);
