#include "Node.hpp"
#include <vector>
#include <random>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <ranges>
#include "DecisionTreeO.hpp"


DecisionTree::DecisionTree(const int max_depth_, const int min_samples_, const unsigned int seed)
    : max_depth(max_depth_), min_samples(min_samples_), gen(seed) {
}

double DecisionTree::gini_from_counts(const std::unordered_map<int, int> &counts, const size_t total) {
    if (total == 0) return 0.0;
    double sum = 0.0;
    for (const auto &p: counts) {
        //
        const double pclass = static_cast<double>(p.second) / static_cast<double>(total);
        sum += pclass * pclass;
    }
    return 1.0 - sum;
}

// Gini impurity of labels y
double DecisionTree::gini(const std::vector<int> &y) {
    if (y.empty()) return 0.0;
    std::unordered_map<int, int> counts;
    for (int c: y) counts[c]++;
    return gini_from_counts(counts, y.size());
}

// Chooses the majority label
int DecisionTree::majority_label_from_counts(const std::unordered_map<int, int> &counts) {
    int best_label = 0;
    int best_count = -1;
    for (const auto &[fst, snd]: counts) {
        if (snd > best_count) {
            best_count = snd;
            best_label = fst;
        }
    }
    return best_label;
}

Node *DecisionTree::build(const std::vector<std::vector<double> > &X,
                          const std::vector<int> &y,
                          const std::vector<size_t> &indices,
                          const int depth) {
    // Build label counts for this node
    std::unordered_map<int, int> counts;
    counts.reserve(16);
    for (const size_t idx: indices) counts[y[idx]]++;

    if (depth >= max_depth || indices.size() <= static_cast<size_t>(min_samples) ||
        gini_from_counts(counts, indices.size()) == 0.0) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label_from_counts(counts);
        return leaf;
    }

    const size_t n_features = X[0].size();
    std::uniform_int_distribution feature_dist(0, static_cast<int>(n_features) - 1);

    int best_f = -1;
    double best_t = 0.0;
    double best_score = std::numeric_limits<double>::infinity();

    const int n_try = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(n_features))));
    // For each candidate feature chosen randomly
    for (int k = 0; k < n_try; ++k) {
        const int f = feature_dist(gen);

        // collect (value, index) for current node only (no copying full rows)
        std::vector<std::pair<double, size_t> > vals;
        vals.reserve(indices.size());
        for (size_t idx: indices) vals.emplace_back(X[idx][f], idx);

        std::ranges::sort(vals,
                          [](const auto &a, const auto &b) { return a.first < b.first; });

        // compute prefix counts for gini efficiently
        std::unordered_map<int, int> left_counts;
        left_counts.reserve(16);
        std::unordered_map<int, int> right_counts = counts; // copy
        size_t left_n = 0;

        // iterate potential split positions (unique values)
        for (size_t i = 1; i < vals.size(); ++i) {
            // move sample i-1 to left
            int label_i_1 = y[vals[i - 1].second];
            left_counts[label_i_1] += 1;
            right_counts[label_i_1] -= 1;
            ++left_n;

            // skip identical values for threshold
            if (vals[i].first == vals[i - 1].first) continue;

            const double t = 0.5 * (vals[i].first + vals[i - 1].first);

            const double gL = gini_from_counts(left_counts, left_n);
            const double gR = gini_from_counts(right_counts, vals.size() - left_n);
            const double wL = static_cast<double>(left_n) / static_cast<double>(vals.size());
            const double wR = 1.0 - wL;
            if (const double score = wL * gL + wR * gR; score < best_score) {
                best_score = score;
                best_f = f;
                best_t = t;
            }
        }
    }

    if (best_f == -1) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label_from_counts(counts);
        return leaf;
    }

    // Partition indices into left/right without copying rows
    std::vector<size_t> left_idx;
    left_idx.reserve(indices.size() / 2);
    std::vector<size_t> right_idx;
    right_idx.reserve(indices.size() / 2);
    for (size_t idx: indices) {
        if (X[idx][best_f] <= best_t) left_idx.push_back(idx);
        else right_idx.push_back(idx);
    }

    if (left_idx.empty() || right_idx.empty()) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label_from_counts(counts);
        return leaf;
    }

    const auto node = new Node();
    node->feature = best_f;
    node->threshold = best_t;
    node->left = build(X, y, left_idx, depth + 1);
    node->right = build(X, y, right_idx, depth + 1);
    return node;
}

void DecisionTree::fit(const std::vector<std::vector<double> > &X,
                       const std::vector<int> &y,
                       const std::vector<size_t> &indices) {
    root = build(X, y, indices, 0);
}

int DecisionTree::predict_one(const Node *node, const std::vector<double> &x) {
    if (node->is_leaf) return node->label;
    if (x[node->feature] <= node->threshold) return predict_one(node->left, x);
    return predict_one(node->right, x);
}

int DecisionTree::predict(const std::vector<double> &x) const {
    return predict_one(root, x);
}
