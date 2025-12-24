#include "RandomForestIndexed.hpp"
#include "DecisionTreeIndexed.hpp"

#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

#include "FFNodes.hpp"
#include "DatasetHelper.hpp"
using namespace std;
constexpr bool verbose = false;

// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), seed(seed) {
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, 2, 1, n_classes, seed + i); // each tree gets unique deterministic seed
}

// Train forest with bootstrap sampling (index-based, no copies)
long VersatileRandomForest::fit(const std::vector<std::vector<double> > &X,
                                const std::vector<int> &y, unsigned int parallelMode) {
    // Flat ver
    // Create a flat column-major array from the row-major data
    std::vector<double> X_flat = DatasetHelper::transpose_flat(X);

    // Construct the flat column-major view
    const ColMajorViewFlat Xc{X_flat.data(), X.size(), X[0].size()};

    const size_t size = X.size();
    chrono::time_point<chrono::system_clock, chrono::system_clock::duration> total_start;

    switch (parallelMode) {
        case 1: {
            // OpenMP version
            total_start = std::chrono::high_resolution_clock::now();
            // Parallel loop over trees
#pragma omp parallel default(none) shared(Xc,y, verbose, cout, size, seed)
            {

                // Thread local variables to reuse in loop
                std::mt19937 rng(seed);
                std::uniform_int_distribution<size_t> dist(0, size - 1);
                std::vector<size_t> bootstrap_indices(size);

#pragma omp for schedule(static)
                for (size_t i = 0; i < static_cast<size_t>(n_trees); ++i) {
                    // Create a private RNG per tree to ensure deterministic, thread-safe bootstrap
                    rng.seed(seed + i);
                    // Bootstrap indices
                    // ReSharper disable once CppDFALoopConditionNotUpdated
                    for (size_t j = 0; j < size; ++j)
                        bootstrap_indices[j] = dist(rng);

                    // Fit the tree
                    trees[i].fit(Xc, y, bootstrap_indices);
                }
            }
            break;
        }
        case 2: {
            // FastFlow version
            ssize_t n_workers = ff_numCores();
            // Define workers
            vector<ff_node *> workers(n_workers);
            ranges::generate(workers, [] { return new TreeWorker(); });

            // Setup Farm with Emitter
            ff_farm farm(workers);

            // Use explicit mapping for pinning
            farm.no_mapping();
            total_start = std::chrono::high_resolution_clock::now();
            farm.add_emitter(new TreeBuildEmitter(trees, Xc, y, seed));
            farm.run_and_wait_end();
            break;
        }
        default: {
            // Sequential version
            total_start = chrono::high_resolution_clock::now();
            // Distribution to use for bootstrap sampling
            uniform_int_distribution<size_t> dist(0, X.size() - 1);
            std::mt19937 rng(seed);
            // Fit each tree on a bootstrap sample
            std::vector<size_t> bootstrap_idx(size);
            for (size_t i = 0; i < static_cast<size_t>(n_trees); ++i) {
                rng.seed(seed + i);
                // Generates indices to create the bootstrap sample to build the tree
                for (size_t j = 0; j < size; ++j)
                    bootstrap_idx[j] = dist(rng);
                trees[i].fit(Xc, y, bootstrap_idx);
            }
        }
    }

    const auto total_end = std::chrono::high_resolution_clock::now();
    const long total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    std::cout << "RandomForest fit() total time: "
            << total_time
            << " us\n";
    return total_time;
}

// Batch prediction
vector<int> VersatileRandomForest::predict_batch(const vector<vector<double> > &X, unsigned int parallelMode) const {
    const size_t N = X.size();
    vector<int> predictions(N);
    chrono::time_point<chrono::system_clock> start;
    switch (parallelMode) {
        case 1: {
            start = chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) default(none) shared(X, predictions, N)
            for (size_t i = 0; i < N; ++i)
                predictions[i] = predict(X[i]);
            break;
        }
        case 2: {
            const size_t nCores = ff_numCores();
            size_t chunk_size = std::min((X.size() + nCores - 1) / nCores, 64UL);
            cout << "Predicting " << X.size() << " samples using " << nCores << " cores with chunk size " << chunk_size
                    << "." << endl;
            // Define Workers
            std::vector<ff_node *> workers(nCores);
            ranges::generate(workers, [] { return new PredictWorker(); });

            // Setup Farm with Emitter
            ff_farm farm(workers);
            farm.no_mapping();
            farm.add_emitter(new PredictEmitter(*this, X, predictions, chunk_size));

            // Run farm and time it
            start = std::chrono::high_resolution_clock::now();
            farm.run_and_wait_end();
            break;
        }
        default: {
            start = chrono::high_resolution_clock::now();
            for (size_t i = 0; i < N; ++i)
                predictions[i] = predict(X[i]);
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();

    cout << "RandomForest predict_batch() total time: "
            << chrono::duration_cast<chrono::microseconds>(end - start).count()
            << " us" << endl;
    return predictions;
}
