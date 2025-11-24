#include "DecisionTree.hpp"
#include "Node.hpp"
#include <cmath>
#include <ranges>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <iostream>
using namespace std;

// Constructor
DecisionTree::DecisionTree(const int max_depth_, const int min_samples_, const unsigned int seed)
    : max_depth(max_depth_), min_samples(min_samples_), root(nullptr), gen(seed) {}

// Multi-class Gini impurity
// Gini is defined as 1 - sum(p_i^2) for each class i
double DecisionTree::gini(const vector<int>& y) {
    if (y.empty()) return 0.0;
    // Count class occurrences
    unordered_map<int,int> counts;
    for (int c : y) counts[c]++;
    // Compute Gini impurity
    double sum = 0.0;
    const auto n = static_cast<double>(y.size());
    // Sum of squared class probabilities
    for (const auto &count: counts | views::values) {
        const double p = count / n;
        sum += p * p;
    }
    return 1.0 - sum;
}

// Split quality (score) using Gini impurity
// Used to evaluate potential splits
double DecisionTree::split_score(const vector<vector<double>>& X, const vector<int>& y, const int f, const double t) {
    vector<int> left_y, right_y;
    for (size_t i = 0; i < X.size(); i++)
        (X[i][f] <= t ? left_y : right_y).push_back(y[i]);

    const double gL = gini(left_y);
    const double gR = gini(right_y);
    const double wL = static_cast<double>(left_y.size()) / static_cast<double>(X.size());
    const double wR = 1.0 - wL;
    return wL * gL + wR * gR;
}

// Majority label
// Returns the most common class label in y
int DecisionTree::majority_label(const vector<int>& y) {
    unordered_map<int,int> counts;
    // Count class occurrences
    for (int c : y) counts[c]++;
    // Find label with maximum count
    return ranges::max_element(counts,
                               [](const auto& a, const auto& b){ return a.second < b.second; })->first;
}

// Recursive tree build
// Constructs the decision tree recursively
// Splits data based on best gini impurity of a feature, selected randomly
// Then calls itself for left and right child nodes, until stopping criteria met
// Measures and logs time taken for building the tree
Node* DecisionTree::build(const vector<vector<double>>& X, const vector<int>& y, const int depth) {
    const auto t_start = chrono::high_resolution_clock::now();

    // Stopping criteria
    if (depth >= max_depth || y.size() <= static_cast<size_t>(min_samples) || gini(y) == 0) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label(y);
        return leaf;
    }

    const size_t n_features = X[0].size();
    uniform_int_distribution feature_dist(0, static_cast<int>(n_features) - 1);

    // Find best split
    int best_f = -1;
    double best_t = 0;
    double best_g = numeric_limits<double>::infinity();

    const int n_try = max(1, static_cast<int>(sqrt(n_features)));
    // Evaluate potential splits using random feature selection
    for (int k = 0; k < n_try; k++) {
        const int f = feature_dist(gen);
        vector<double> vals;
        for (auto& r : X) vals.push_back(r[f]);
        ranges::sort(vals);

        // For each unique threshold candidate, evaluate split score and track best
        for (size_t i = 1; i < vals.size(); i++) {
            const double t = 0.5 * (vals[i] + vals[i-1]);
            if (const double g = split_score(X, y, f, t); g < best_g) {
                best_g = g;
                best_f = f;
                best_t = t;
            }
        }
    }

    // If no valid split found, create leaf with majority label
    if (best_f == -1) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label(y);
        return leaf;
    }

    // Otherwise split into left and right subsets to build the next two nodes
    vector<vector<double>> XL, XR;
    vector<int> yL, yR;
    for (size_t i = 0; i < X.size(); i++) {
        if (X[i][best_f] <= best_t) {
            XL.push_back(X[i]); yL.push_back(y[i]);
        } else {
            XR.push_back(X[i]); yR.push_back(y[i]);
        }
    }

    // If either side is empty, create leaf with majority label
    if (XL.empty() || XR.empty()) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label(y);
        return leaf;
    }

    // Call build recursively for left and right child nodes
    const auto node = new Node();
    node->feature = best_f;
    node->threshold = best_t;
    node->left = build(XL, yL, depth+1);
    node->right = build(XR, yR, depth+1);

    const auto t_end = chrono::high_resolution_clock::now();
    const auto duration = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();
    if (depth == 0) {
        cout << "[Timing] Tree build completed in " << duration << " ms" << endl;
    }

    return node;
}

// Predict single sample
int DecisionTree::predict_one(const Node* node, const vector<double>& x) {
    // Base case: leaf node -> return label
    if (node->is_leaf) return node->label;
    // Recursive case: traverse left or right child based on feature threshold
    return predict_one(x[node->feature] <= node->threshold ? node->left : node->right, x);
}

// Fit
// Build the decision tree from training data
// Measures and log time elapsed
void DecisionTree::fit(const vector<vector<double>>& X, const vector<int>& y) {
    const auto start = chrono::high_resolution_clock::now();
    root = build(X, y, 0);
    const auto end = chrono::high_resolution_clock::now();
    cout << "[Timing] fit() total time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
}

// Predict a single sample
int DecisionTree::predict(const vector<double>& x) const {
    return predict_one(root, x);
}

// Batch prediction, with timer logging
vector<int> DecisionTree::predict_batch(const vector<vector<double>>& X) const {
    const auto start = chrono::high_resolution_clock::now();
    vector<int> preds;
    preds.reserve(X.size());
    for (auto& row : X)
        preds.push_back(predict_one(root, row));
    const auto end = chrono::high_resolution_clock::now();
    cout << "[Timing] predict() batch time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
    return preds;
}
