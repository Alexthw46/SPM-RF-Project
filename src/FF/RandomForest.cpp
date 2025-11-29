#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <unordered_map>
#include <ranges>

// ReSharper disable once CppUnusedIncludeDirective
#include <ff/ff.hpp>
#include <ff/farm.hpp>

#include "CSVLoader.hpp"
#include "DecisionTreeIndexed.hpp"
#include "RandomForestIndexed.hpp"
#include "FFNodes.hpp"

using namespace std;
using namespace ff;

// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), gen(seed) {
    int min_samples = 2;
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, min_samples, seed + i); // each tree gets unique deterministic seed
}

// Create bootstrap sample indices for a single tree.
inline void create_bootstrap_indexes(
    const std::vector<std::vector<double> > &X,
    std::uniform_int_distribution<size_t> &dist,
    std::mt19937 &rng,
    std::vector<size_t> &bootstrap_idx) {
    const size_t n = X.size();
    bootstrap_idx.resize(n);

    for (size_t i = 0; i < n; i++)
        bootstrap_idx[i] = dist(rng);
}

// Train forest with bootstrap sampling (index-based, no copies)
// FastFlow version
void RandomForest::fit(const vector<vector<double>> &X, const vector<int> &y) {
    ssize_t n_workers = ff_numCores();
    // Define workers
    vector<ff_node *> workers(n_workers);
    ranges::generate(workers, [] { return new TreeWorker2(); });

    // Create a flat column-major array from the row-major data
    std::vector<double> X_flat = CSVLoader::transpose_flat(X);

    // Construct the flat column-major view
    const ColMajorViewFlat Xc{X_flat.data(), X.size(), X[0].size()};

    uint64_t seed_base = gen();

    // Setup Farm with Emitter
    ff_farm farm(workers);
    farm.add_emitter(new TreeBuildEmitter2(trees, Xc, y, n_workers, seed_base));

    // Run farm and time it
    const auto total_start = chrono::high_resolution_clock::now();
    farm.run_and_wait_end();
    const auto total_end = chrono::high_resolution_clock::now();
    cout << "All trees built in parallel using " << n_workers << " workers." << endl;

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

std::vector<int> RandomForest::predict_batch(const std::vector<std::vector<double> > &X) const {
    std::vector<int> predictions(X.size());
    const size_t nCores = ff_numCores();

    // Define Workers
    std::vector<ff_node *> workers(nCores);
    ranges::generate(workers, [] { return new PredictWorker(); });

    // Setup Farm with Emitter
    ff_farm farm(workers);
    farm.add_emitter(new PredictEmitter(*this, X, predictions, 50));

    // Run farm and time it
    const auto start = std::chrono::high_resolution_clock::now();
    farm.run_and_wait_end();
    const auto end = std::chrono::high_resolution_clock::now();

    std::cout << "[Timing] RandomForest predict_batch_parallel() total time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms" << std::endl;

    return predictions;
}
