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
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), seed(seed) {
    int min_samples = 2;
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, min_samples, 1, n_classes, seed + i); // each tree gets unique deterministic seed
}

// FastFlow version
void RandomForest::fit(const vector<vector<double> > &X, const vector<int> &y) {
    ssize_t n_workers = ff_numCores();
    // Define workers
    vector<ff_node *> workers(n_workers);

    ranges::generate(workers, [] { return new TreeWorker(); });

    // Create a flat column-major array from the row-major data
    std::vector<double> X_flat = CSVLoader::transpose_flat(X);

    // Construct the flat column-major view
    const ColMajorViewFlat Xc{X_flat.data(), X.size(), X[0].size()};

    // Setup Farm with Emitter
    ff_farm farm(workers);

    // Use explicit mapping for pinning
    farm.no_mapping();

    farm.add_emitter(new TreeBuildEmitter(trees, Xc, y, seed));

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
    size_t chunk_size = (X.size() + nCores - 1) / nCores; // ceiling division
    // Avoid chunks too small
    if (chunk_size < 50)
        chunk_size = 50;
    // cout << "Predicting " << X.size() << " samples using " << nCores << " cores with chunk size " << chunk_size << "." << endl;
    // Define Workers
    std::vector<ff_node *> workers(nCores);
    ranges::generate(workers, [] { return new PredictWorker(); });

    // Setup Farm with Emitter
    ff_farm farm(workers);
    farm.no_mapping();
    farm.add_emitter(new PredictEmitter(*this, X, predictions, chunk_size));

    // Run farm and time it
    const auto start = std::chrono::high_resolution_clock::now();
    farm.run_and_wait_end();
    const auto end = std::chrono::high_resolution_clock::now();

    std::cout << "[Timing] RandomForest predict_batch() total time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms" << std::endl;

    return predictions;
}
