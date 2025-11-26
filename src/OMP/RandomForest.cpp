#include "RandomForestIndexed.hpp"
#include "DecisionTreeIndexed.hpp"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <iostream>

#include "CSVLoader.hpp"
using namespace std;
constexpr bool verbose = false;

// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), gen(seed) {
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, 2, seed + i); // each tree gets unique deterministic seed
}

// Train forest with bootstrap sampling (index-based, no copies)
void RandomForest::fit(const std::vector<std::vector<double> > &X,
                       const std::vector<int> &y) {
    const auto total_start = std::chrono::high_resolution_clock::now();

    // Flat ver
    // Create a flat column-major array from the row-major data
    std::vector<double> X_flat = CSVLoader::transpose_flat(X);

    // Construct the flat column-major view
    const ColMajorViewFlat Xc{X_flat.data(), X.size(), X[0].size()};

    std::vector<std::vector<size_t> > bootstrap_indices(n_trees);
    const size_t size = X.size();

    // Parallel loop over trees
#pragma omp parallel for schedule(static) default(none) shared(Xc,y, bootstrap_indices, verbose, cout) firstprivate(size)
    for (size_t i = 0; i < static_cast<size_t>(n_trees); ++i) {
        // Create a private RNG per tree to ensure deterministic, thread-safe bootstrap
        std::mt19937 rng(gen() + i);
        std::uniform_int_distribution<size_t> dist(0, size - 1);

        // Bootstrap indices
        bootstrap_indices[i].resize(size);
        for (size_t j = 0; j < size; ++j)
            bootstrap_indices[i][j] = dist(rng);

        // Fit the tree
        const auto t_start = std::chrono::high_resolution_clock::now();
        trees[i].fit(Xc, y, bootstrap_indices[i]);
        const auto t_end = std::chrono::high_resolution_clock::now();
        if (verbose)
#pragma omp critical
            std::cout << "[Timing] Tree " << i << " trained in "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
                    << " ns\n";
    }

    const auto total_end = std::chrono::high_resolution_clock::now();
    std::cout << "[Timing] RandomForest fit() total time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count()
            << " ms\n";
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
    const size_t N = X.size();
    vector<int> predictions(N);
    const auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static) default(none) shared(X, predictions, N)
    for (size_t i = 0; i < N; ++i)
        predictions[i] = predict(X[i]);

    const auto end = chrono::high_resolution_clock::now();
    cout << "[Timing] RandomForest predict_batch() total time: "
            << chrono::duration_cast<chrono::milliseconds>(end - start).count()
            << " ms" << endl;
    return predictions;
}
