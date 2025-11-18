#include "RandomForest.hpp"
#include "DecisionTree.hpp"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <iostream>
using namespace std;

RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), gen(seed) {
    // Initialize trees
    for (int i = 0; i < n_t; i++)
        trees.emplace_back(max_depth, 2, seed + i);
}

// Train forest with bootstrap samples
void RandomForest::fit(const vector<vector<double> > &X, const vector<int> &y) {
    const auto total_start = chrono::high_resolution_clock::now();

    // Distribution to use for bootstrap sampling
    uniform_int_distribution<size_t> dist(0, X.size() - 1);

    // Fit each tree on a bootstrap sample
    for (size_t i = 0; i < trees.size(); i++) {
        auto &t = trees[i];
        vector<vector<double> > Xb;
        vector<int> yb;
        Xb.reserve(X.size());
        yb.reserve(X.size());

        // Generates an index to create the bootstrap sample to build the tree
        for (size_t j = 0; j < X.size(); j++) {
            const size_t idx = dist(gen);
            Xb.push_back(X[idx]);
            yb.push_back(y[idx]);
        }

        // Time the fitting of each tree
        auto t_start = chrono::high_resolution_clock::now();
        t.fit(Xb, yb);
        auto t_end = chrono::high_resolution_clock::now();

        cout << "[Timing] Tree " << i << " trained in "
                << chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count()
                << " ms" << endl;
    }

    const auto total_end = chrono::high_resolution_clock::now();
    cout << "[Timing] RandomForest fit() total time: "
            << chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count()
            << " ms" << endl;
}

// Predict for one sample
int RandomForest::predict(const vector<double> &x) const {
    unordered_map<int, int> vote_count;
    // Collect votes from each tree
    for (const auto &t: trees) {
        int p = t.predict(x);
        vote_count[p]++;
    }
    // Return label with most votes
    return ranges::max_element(vote_count.begin(), vote_count.end(),
                               [](const auto &a, const auto &b) { return a.second < b.second; })->first;
}

// Batch prediction
vector<int> RandomForest::predict_batch(const vector<vector<double> > &X) const {
    const auto start = chrono::high_resolution_clock::now();
    vector<int> predictions;
    predictions.reserve(X.size());
    for (auto &row: X)
        predictions.push_back(predict(row));
    const auto end = chrono::high_resolution_clock::now();
    cout << "[Timing] RandomForest predict_batch() total time: "
            << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
            << " ns" << endl;
    return predictions;
}
