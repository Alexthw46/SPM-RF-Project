#include "RandomForestIndexed.hpp"
#include "DecisionTreeIndexed.hpp"

#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "DatasetHelper.hpp"

using namespace std;

constexpr bool verbose = false;

// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), seed(seed) {
    int rank, n_ranks;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const size_t trees_per_rank = (n_trees + n_ranks - 1) / n_ranks;
    const size_t start_tree = rank * trees_per_rank;

    const size_t local_trees = min(rank * trees_per_rank + trees_per_rank, static_cast<size_t>(n_trees)) - start_tree;
    trees.reserve(local_trees);
    // Initialize trees
    for (size_t i = 0; i < local_trees; i++)
        // each tree gets unique, deterministic, seed
        trees.emplace_back(max_depth, 2, 1, n_classes, seed + start_tree + i);
}

long RandomForest::fit(const vector<vector<double> > &X,
                       const vector<int> &y) {
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const size_t trees_per_rank = (n_trees + n_ranks - 1) / n_ranks;
    const size_t start_tree = rank * trees_per_rank;
    const size_t local_trees = trees.size();
    const size_t size = X.size();

    // Create a flat column-major array from the row-major data
    std::vector<double> X_flat = DatasetHelper::transpose_flat(X);

    // Construct the flat column-major view
    const ColMajorViewFlat Xc{X_flat.data(), X.size(), X[0].size()};

    const auto total_start = chrono::high_resolution_clock::now();
#pragma omp parallel default(none) shared(Xc, trees, y, start_tree, local_trees, \
    size, rank, seed, cout)
    {
        // Bootstrap variables are thread-local and reused to reduce allocations, since size is constant
        std::vector<size_t> bootstrap_indices(size);
        std::uniform_int_distribution<size_t> dist(0, size - 1);
        std::mt19937 rng(seed);
#pragma omp for schedule(static)
        // ReSharper disable once CppDFALoopConditionNotUpdated
        for (size_t idx = 0; idx < local_trees; ++idx) {
            rng.seed(seed + start_tree + idx);
            // Generates indices to create the bootstrap sample to build the tree
            // ReSharper disable once CppDFALoopConditionNotUpdated
            for (size_t j = 0; j < size; ++j)
                bootstrap_indices[j] = dist(rng);
            trees[idx].fit(Xc, y, bootstrap_indices); // Fit tree with bootstrap sample
        }
    }

    // Timer for the current rank
    const auto rank_end = chrono::high_resolution_clock::now();
    const long rank_time = chrono::duration_cast<chrono::microseconds>(rank_end - total_start).count();

    if (n_ranks > 1) // Needed to get the correct total time, only if there are multiple ranks
        MPI_Barrier(MPI_COMM_WORLD);

    // Total timer including barriers
    const auto total_end = chrono::high_resolution_clock::now();
    const long total_time = chrono::duration_cast<chrono::microseconds>(total_end - total_start).count();
    if (rank == 0)
        cout << "RandomForest MPI fit() total time: "
                << total_time
                << " us\n"
                << "Rank 0 training time before barrier: " << rank_time << " us" << endl;
    return rank_time;
}

// Batch prediction
std::vector<int> RandomForest::predict_batch(const std::vector<std::vector<double> > &X) const {
    // Data is fully replicated on each rank, but each rank only has a subset of trees
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    const size_t N = X.size();
    const size_t trees_per_rank = (n_trees + n_ranks - 1) / n_ranks;
    const size_t local_trees = trees.size();
    if (verbose) {
        const size_t start_tree = rank * trees_per_rank;
        const size_t end_tree = min(start_tree + trees_per_rank, static_cast<size_t>(n_trees));
        cout << "[Rank " << rank << "] Predicting with trees " << start_tree << " to " << end_tree - 1 << endl;
    }
    const auto startT = chrono::high_resolution_clock::now();
    std::vector local_votes(N * n_classes, 0); // Local votes for this rank
#pragma omp parallel for schedule(static) default(none) shared(X, local_votes, local_trees, trees, N, n_classes\
)
    for (size_t i = 0; i < N; ++i) {
        for (size_t t = 0; t < local_trees; ++t) {
            const int p = trees[t].predict(X[i]); // Predict class for sample i with tree t, p in [0, n_classes-1]
            local_votes[i * n_classes + p]++;
            // Unique index for each (i, p) pair, false sharing mitigated by omp static schedule
        }
    }

    // Reduce votes on rank 0
    std::vector<int> global_votes;
    if (rank == 0) // Only rank 0 needs to store the global votes
        global_votes.resize(N * n_classes);

    MPI_Reduce(local_votes.data(),
               rank == 0 ? global_votes.data() : nullptr,
               static_cast<int>(N * n_classes),
               MPI_INT,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    // Determine final predictions on rank 0
    vector<int> predictions(N);
    if (rank == 0) {
#pragma omp parallel for schedule(static) default(none) shared(global_votes, predictions, N, n_classes)
        for (size_t i = 0; i < N; ++i) {
            int best_class = 0;

            const size_t sample_vote_start_index = i * n_classes;
            int best_count = global_votes[sample_vote_start_index]; // initialized with votes for class 0
            for (int c = 1; c < n_classes; ++c) {
                // start from class 1
                if (const int count = global_votes[sample_vote_start_index + c]; count > best_count) {
                    best_count = count;
                    best_class = c;
                }
            }
            predictions[i] = best_class; // Writes to unique index. False sharing mitigated by omp static schedule
        }
        const auto endT = chrono::high_resolution_clock::now();
        cout << "[Timing Rank " << rank << "] RandomForest predict_batch() total time: "
                << chrono::duration_cast<chrono::microseconds>(endT - startT).count()
                << " us" << endl;
    }
    return predictions; // Other ranks return empty vector
}
