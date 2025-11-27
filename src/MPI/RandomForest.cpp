#include "RandomForestIndexed.hpp"
#include "DecisionTreeIndexed.hpp"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <vector>

#include "CSVLoader.hpp"

using namespace std;

constexpr bool verbose = true;

// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), gen(seed) {
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, 2, seed + i); // each tree gets unique deterministic seed
}

void RandomForest::fit(const vector<vector<double> > &X,
                       const vector<int> &y) {
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const size_t trees_per_rank = (n_trees + n_ranks - 1) / n_ranks;
    const size_t start_tree = rank * trees_per_rank;
    const size_t end_tree = min(start_tree + trees_per_rank, static_cast<size_t>(n_trees));
    const size_t size = X.size();

    // Flat ver
    // Create a flat column-major array from the row-major data
    std::vector<double> X_flat = CSVLoader::transpose_flat(X);

    // Construct the flat column-major view
    const ColMajorViewFlat Xc{X_flat.data(), X.size(), X[0].size()};

    std::vector<std::vector<size_t> > bootstrap_indices(end_tree - start_tree);
    const auto total_start = chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) default(none) shared(Xc,y ,bootstrap_indices, verbose, rank, start_tree, \
    end_tree, cout) firstprivate(size)
    for (size_t idx = start_tree; idx < end_tree; ++idx) {
        const size_t local_idx = idx - start_tree;

        std::mt19937 rng(gen() + idx);
        std::uniform_int_distribution<size_t> dist(0, size - 1);

        bootstrap_indices[local_idx].resize(size);
        for (size_t j = 0; j < size; ++j)
            bootstrap_indices[local_idx][j] = dist(rng);

        const auto t_start = chrono::high_resolution_clock::now();
        trees[idx].fit(Xc, y, bootstrap_indices[local_idx]);
        const auto t_end = chrono::high_resolution_clock::now();

        if (verbose)
#pragma omp critical
            cout << "[Rank " << rank << "] Tree " << idx
                    << " trained in "
                    << chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count()
                    << " ns\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto total_end = chrono::high_resolution_clock::now();

    if (rank == 0)
        cout << "[Timing] RandomForest MPI fit() total time: "
                << chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count()
                << " ms\n";
}

// Predict for one sample
int RandomForest::predict(const vector<double> &x) const {
    unordered_map<int, int> vote_count;
    // Collect votes from each tree
    for (const auto &t: trees) {
        if (!t.root)
            continue;
        int p = t.predict(x);
        vote_count[p]++;
    } // Return label with most votes
    return ranges::max_element(vote_count.begin(), vote_count.end(),
                               [](const auto &a, const auto &b) { return a.second < b.second; })->first;
}

// Batch prediction
vector<int> RandomForest::predict_batch(const vector<vector<double> > &X) const {
    int rank, n_ranks;
    const auto startT = chrono::high_resolution_clock::now();

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
        const size_t s = static_cast<size_t>(r) * chunk_size;
        const size_t e = min(s + chunk_size, N);
        counts[r] = static_cast<int>(e - s);
        displs[r] = static_cast<int>(s);
    }

    MPI_Gatherv(local_predictions.data(), local_predictions.size(), MPI_INT,
                predictions.data(), counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    const auto endT = chrono::high_resolution_clock::now();
    cout << "[Timing] RandomForest predict_batch() total time: "
            << chrono::duration_cast<chrono::milliseconds>(endT - startT).count()
            << " ms" << endl;
    return predictions; // valid only on rank 0
}

void RandomForestReplicated::gather_all_trees(MPI_Comm comm) {
    int rank, n_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);
}
