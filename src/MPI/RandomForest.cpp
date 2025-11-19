#include "RandomForestIndexed.hpp"
#include "DecisionTreeIndexed.hpp"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <iostream>

#include <mpi.h>
#include <vector>

using namespace std;

// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), gen(seed) {
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, 2, seed + i); // each tree gets unique deterministic seed
}

void RandomForest::fit(const vector<vector<double>> &X,
                           const vector<int> &y)
{
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const size_t trees_per_rank = (n_trees + n_ranks - 1) / n_ranks;
    const size_t start_tree = rank * trees_per_rank;
    const size_t end_tree = min(start_tree + trees_per_rank, static_cast<size_t>(n_trees));

    uniform_int_distribution<size_t> dist(0, X.size() - 1);

    const auto total_start = chrono::high_resolution_clock::now();

    for (size_t i = start_tree; i < end_tree; ++i) {
        vector<size_t> bootstrap_idx(X.size());
        for (size_t j = 0; j < X.size(); ++j)
            bootstrap_idx[j] = dist(gen);

        const auto t_start = chrono::high_resolution_clock::now();
        trees[i].fit(X, y, bootstrap_idx);
        const auto t_end = chrono::high_resolution_clock::now();

        cout << "[Rank " << rank << "] Tree " << i
                  << " trained in "
                  << chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count()
                  << " ns\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto total_end = chrono::high_resolution_clock::now();

    if (rank == 0)
        cout << "[Timing] RandomForest MPI fit() total time: "
                  << chrono::duration_cast<chrono::nanoseconds>(total_end - total_start).count()
                  << " ns\n";
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
vector<int> RandomForest::predict_batch(const vector<vector<double>> &X) const {
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const size_t N = X.size();
    const size_t chunk_size = (N + n_ranks - 1) / n_ranks;
    const size_t start = rank * chunk_size;
    const size_t end = min(start + chunk_size, N);

    vector<int> local_predictions(end - start);
    for (size_t i = start; i < end; ++i)
        local_predictions[i - start] = predict(X[i]);

    vector<int> predictions;
    if (rank == 0)
        predictions.resize(N);

    // Gather results from all ranks
    vector<int> counts(n_ranks);
    vector<int> displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const size_t s = r * chunk_size;
        const size_t e = min(s + chunk_size, N);
        counts[r] = e - s;
        displs[r] = s;
    }

    MPI_Gatherv(local_predictions.data(), local_predictions.size(), MPI_INT,
                predictions.data(), counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    return predictions; // valid only on rank 0
}
