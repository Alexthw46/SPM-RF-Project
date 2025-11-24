#include "RandomForestIndexed.hpp"
#include "DecisionTreeIndexed.hpp"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <iostream>

#include "CSVLoader.hpp"
using namespace std;

// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), gen(seed) {
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, 2, seed + i); // each tree gets unique deterministic seed
}

// Train forest with bootstrap sampling (index-based, no copies)
void RandomForest::fit(const vector<vector<double> > &X, const vector<int> &y) {
    const auto total_start = chrono::high_resolution_clock::now();


    // Distribution to use for bootstrap sampling
    uniform_int_distribution<size_t> dist(0, X.size() - 1);

    // Transpose the sample-feature matrix once
    const ColMajorView Xc = {CSVLoader::transpose(X)};

    // Fit each tree on a bootstrap sample
    for (size_t i = 0; i < trees.size(); i++) {
        auto &t = trees[i];

        // Generates indices to create the bootstrap sample to build the tree
        // (no copies of data)
        vector<size_t> bootstrap_idx;
        bootstrap_idx.reserve(X.size());
        for (size_t j = 0; j < X.size(); j++)
            bootstrap_idx.push_back(dist(gen));

        // Time the fitting of each tree
        const auto t_start = chrono::high_resolution_clock::now();
        t.fit(Xc, y, bootstrap_idx);
        const auto t_end = chrono::high_resolution_clock::now();

        cout << "[Timing] Tree " << i << " trained in "
                << chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count()
                << " ns" << endl;
    }
    cout << "All trees built Sequentially." << endl;

    const auto total_end = chrono::high_resolution_clock::now();
    cout << "[Timing] RandomForest fit() total time: "
            << chrono::duration_cast<chrono::nanoseconds>(total_end - total_start).count()
            << " ns" << endl;
}

// Predict for one sample
int RandomForest::predict(const vector<double> &x) const {
    unordered_map<int, int> vote_count;
    // Collect votes from each tree
    for (const auto &t: trees) {
        int p = t.predict(x);
        vote_count[p]++;
    } // Return label with most votes
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
