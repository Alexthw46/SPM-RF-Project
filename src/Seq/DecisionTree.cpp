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
double DecisionTree::gini(const vector<int>& y) {
    if (y.empty()) return 0.0;
    unordered_map<int,int> counts;
    for (int c : y) counts[c]++;
    double sum = 0.0;
    const auto n = static_cast<double>(y.size());
    for (const auto &count: counts | views::values) {
        const double p = count / n;
        sum += p * p;
    }
    return 1.0 - sum;
}

// Split quality
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
int DecisionTree::majority_label(const vector<int>& y) {
    unordered_map<int,int> counts;
    for (int c : y) counts[c]++;
    return ranges::max_element(counts,
                               [](const auto& a, const auto& b){ return a.second < b.second; })->first;
}

// Recursive tree build
Node* DecisionTree::build(const vector<vector<double>>& X, const vector<int>& y, const int depth) {
    const auto t_start = chrono::high_resolution_clock::now();

    if (depth >= max_depth || y.size() <= min_samples || gini(y) == 0) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label(y);
        return leaf;
    }

    const size_t n_features = X[0].size();
    uniform_int_distribution<int> feature_dist(0, static_cast<int>(n_features) - 1);

    int best_f = -1;
    double best_t = 0;
    double best_g = numeric_limits<double>::max();

    const int n_try = max(1, static_cast<int>(sqrt(n_features)));
    for (int k = 0; k < n_try; k++) {
        const int f = feature_dist(gen);
        vector<double> vals;
        for (auto& r : X) vals.push_back(r[f]);
        ranges::sort(vals);

        for (size_t i = 1; i < vals.size(); i++) {
            const double t = 0.5 * (vals[i] + vals[i-1]);
            if (const double g = split_score(X, y, f, t); g < best_g) {
                best_g = g;
                best_f = f;
                best_t = t;
            }
        }
    }

    if (best_f == -1) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label(y);
        return leaf;
    }

    vector<vector<double>> XL, XR;
    vector<int> yL, yR;
    for (size_t i = 0; i < X.size(); i++) {
        if (X[i][best_f] <= best_t) {
            XL.push_back(X[i]); yL.push_back(y[i]);
        } else {
            XR.push_back(X[i]); yR.push_back(y[i]);
        }
    }

    if (XL.empty() || XR.empty()) {
        const auto leaf = new Node();
        leaf->is_leaf = true;
        leaf->label = majority_label(y);
        return leaf;
    }

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
    if (node->is_leaf) return node->label;
    if (x[node->feature] <= node->threshold)
        return predict_one(node->left, x);
    else
        return predict_one(node->right, x);
}

// Fit
void DecisionTree::fit(const vector<vector<double>>& X, const vector<int>& y) {
    const auto start = chrono::high_resolution_clock::now();
    root = build(X, y, 0);
    const auto end = chrono::high_resolution_clock::now();
    cout << "[Timing] fit() total time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;
}

// Predict
int DecisionTree::predict(const vector<double>& x) const {
    return predict_one(root, x);
}

// Batch prediction timing helper (optional)
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
