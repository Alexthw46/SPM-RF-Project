#include "RandomForestIndexed.hpp"
#include "DecisionTreeIndexed.hpp"

#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <vector>

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

long RandomForest::fit(const vector<vector<double> > &X,
                       const vector<int> &y) {
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const size_t trees_per_rank = (n_trees + n_ranks - 1) / n_ranks;
    const size_t start_tree = rank * trees_per_rank;
    const size_t end_tree = min(start_tree + trees_per_rank, static_cast<size_t>(n_trees));
    const size_t size = X.size();

    // Create a flat column-major array from the row-major data
    std::vector<double> X_flat = DatasetHelper::transpose_flat(X);

    // Construct the flat column-major view
    const ColMajorViewFlat Xc{X_flat.data(), X.size(), X[0].size()};

    const auto total_start = chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) default(none) shared(Xc, trees, y, start_tree, end_tree,\
    size, rank, seed, cout)
    for (size_t idx = start_tree; idx < end_tree; ++idx) {
        std::mt19937 rng(seed + idx);
        std::uniform_int_distribution<size_t> dist(0, size - 1);

        std::vector<size_t> bootstrap_indices(size);
        for (size_t j = 0; j < size; ++j)
            bootstrap_indices[j] = dist(rng);

        const auto t_start = chrono::high_resolution_clock::now();
        trees[idx].fit(Xc, y, bootstrap_indices);
        const auto t_end = chrono::high_resolution_clock::now();

        if (verbose)
#pragma omp critical
            cout << "[Rank " << rank << "] Tree " << idx
                    << " trained in "
                    << chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count()
                    << " ns\n";
    }

    if (n_ranks > 1)
        MPI_Barrier(MPI_COMM_WORLD);
    const auto total_end = chrono::high_resolution_clock::now();

    const long total_time = chrono::duration_cast<chrono::microseconds>(total_end - total_start).count();
    if (rank == 0)
        cout << "RandomForest MPI fit() total time: "
                << total_time
                << " us\n";
    return total_time;
}

// Batch prediction
std::vector<int> RandomForest::predict_batch(const std::vector<std::vector<double> > &X) const {
    // Data is fully replicated on each rank, but each rank only has a subset of trees
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    const size_t N = X.size();
    const size_t trees_per_rank = (n_trees + n_ranks - 1) / n_ranks;
    const size_t start_tree = rank * trees_per_rank;
    const size_t end_tree = min(start_tree + trees_per_rank, static_cast<size_t>(n_trees));
    if (verbose)
        cout << "[Rank " << rank << "] Predicting with trees " << start_tree << " to " << end_tree - 1 << endl;
    const auto startT = chrono::high_resolution_clock::now();
    std::vector local_votes(N * n_classes, 0);
#pragma omp parallel for schedule(static) default(none) shared(X, local_votes, start_tree, end_tree, trees, N, n_classes)
    for (size_t i = 0; i < N; ++i) {
        for (size_t t = start_tree; t < end_tree; ++t) {
            const int p = trees[t].predict(X[i]);
#pragma omp atomic
            local_votes[i * n_classes + p]++;
        }
    }
    // Reduce votes on rank 0
    std::vector<int> global_votes;
    if (rank == 0)
        global_votes.resize(N * n_classes);

    MPI_Reduce(local_votes.data(),
               rank == 0 ? global_votes.data() : nullptr,
               static_cast<int>(N * n_classes),
               MPI_INT,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);
    // Determine final predictions, only needed on rank 0
    vector<int> predictions(N);
    if (rank == 0) {
#pragma omp parallel for schedule(static) default(none) shared(global_votes, predictions, N, n_classes)
        for (size_t i = 0; i < N; ++i) {
            int best_class = 0;
            int best_count = global_votes[i * n_classes];
            for (int c = 1; c < n_classes; ++c) {
                if (const int count = global_votes[i * n_classes + c]; count > best_count) {
                    best_count = count;
                    best_class = c;
                }
            }
            predictions[i] = best_class;
        }
        const auto endT = chrono::high_resolution_clock::now();
        cout << "[Timing Rank " << rank << "] RandomForest predict_batch() total time: "
                << chrono::duration_cast<chrono::microseconds>(endT - startT).count()
                << " us" << endl;
    }
    return predictions;
}