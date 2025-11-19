#include "Node.hpp"
#include <vector>
#include <random>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <ranges>
#include "DecisionTreeIndexed.hpp"

using namespace std;

DecisionTree::DecisionTree(const int max_depth_, const int min_samples_, const unsigned int seed)
    : max_depth(max_depth_), min_samples(min_samples_), gen(seed) {
}

// Gini impurity from pre-computed class counts
double DecisionTree::gini_from_counts(const unordered_map<int, int> &counts, const size_t n_features) {
    if (n_features == 0) return 0.0;
    double sum = 0.0;
    // Sum of squared class probabilities
    for (const auto &val: counts | views::values) {
        const double pClass = static_cast<double>(val) / static_cast<double>(n_features);
        sum += pClass * pClass;
    }
    return 1.0 - sum;
}

// Gini impurity of labels y
double DecisionTree::gini(const vector<int> &y) {
    if (y.empty()) return 0.0;
    // Count class occurrences
    unordered_map<int, int> counts;
    for (int c: y) counts[c]++;
    return gini_from_counts(counts, y.size());
}

// Chooses the majority label
// Count occurrences are pre-computed
int DecisionTree::majority_label_from_counts(const unordered_map<int, int> &counts) {
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

// Recursive tree build
// Constructs the decision tree recursively
// Splits data based on best gini impurity of a feature, selected randomly
// Then calls itself for left and right child nodes, until stopping criteria met
// Measures and logs time taken for building the tree
Node *DecisionTree::build(const vector<vector<double> > &X,
                          const vector<int> &y,
                          const vector<size_t> &indices,
                          const int depth) {
    // Build label counts for this node
    unordered_map<int, int> counts;
    counts.reserve(16);
    for (const size_t idx: indices) counts[y[idx]]++;

    // Stopping criteria
    if (depth >= max_depth || indices.size() <= static_cast<size_t>(min_samples) || gini_from_counts(
            counts, indices.size()) == 0.0) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label_from_counts(counts);
        return leaf;
    }

    const size_t n_features = X[0].size();
    uniform_int_distribution feature_dist(0, static_cast<int>(n_features) - 1);

    // Find best split
    int best_f = -1;
    double best_t = 0.0;
    double best_score = numeric_limits<double>::infinity();

    const int n_try = max(1, static_cast<int>(sqrt(static_cast<double>(n_features))));
    // For each candidate feature chosen randomly
    for (int k = 0; k < n_try; ++k) {
        const int f = feature_dist(gen);

        // collect (value, index) for current node only (no copying full rows)
        vector<pair<double, size_t> > vals;
        vals.reserve(indices.size());
        for (size_t idx: indices) vals.emplace_back(X[idx][f], idx);

        ranges::sort(vals,
                     [](const auto &a, const auto &b) { return a.first < b.first; });

        // left_counts: counts of labels on the left side of the split
        // right_counts: counts of labels on the right side of the split
        // initialize right to total counts and shrinks as samples move to left
        unordered_map<int, int> left_counts;
        left_counts.reserve(counts.size());
        unordered_map<int, int> right_counts = counts; // copy
        // Keep track of number of samples on left
        size_t left_n = 0;

        // iterate potential split positions (only unique values)
        for (size_t i = 1; i < vals.size(); ++i) {
            // move sample i-1 to left side
            int label_i_1 = y[vals[i - 1].second];
            left_counts[label_i_1] += 1;
            right_counts[label_i_1] -= 1;
            ++left_n;

            // skip identical values for threshold
            if (vals[i].first == vals[i - 1].first) continue;

            // threshold is midpoint between unique values
            const double t = 0.5 * (vals[i].first + vals[i - 1].first);

            // compute Gini efficiently using prefix counts
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

    // If no valid split found, create leaf with majority label
    if (best_f == -1) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label_from_counts(counts);
        return leaf;
    }

    // Partition indices into left/right without copying rows
    vector<size_t> left_idx;
    left_idx.reserve(indices.size() / 2);
    vector<size_t> right_idx;
    right_idx.reserve(indices.size() / 2);
    for (size_t idx: indices) {
        if (X[idx][best_f] <= best_t) left_idx.push_back(idx);
        else right_idx.push_back(idx);
    }

    // If either side is empty, create leaf with majority label
    if (left_idx.empty() || right_idx.empty()) {
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
    node->left = build(X, y, left_idx, depth + 1);
    node->right = build(X, y, right_idx, depth + 1);
    return node;
}

void DecisionTree::fit(const vector<vector<double> > &X,
                       const vector<int> &y,
                       const vector<size_t> &indices) {
    // Start the recursive build from root
    root = build(X, y, indices, 0);
}

int DecisionTree::predict_one(const Node *node, const vector<double> &x) {
    // Base case: leaf node -> return label
    if (node->is_leaf) return node->label;
    // Recursive case: traverse left or right child based on feature threshold
    return predict_one(x[node->feature] <= node->threshold ? node->left : node->right, x);
}

int DecisionTree::predict(const vector<double> &x) const {
    return predict_one(root, x);
}
