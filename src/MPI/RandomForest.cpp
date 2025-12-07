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

    // Flat ver
    // Create a flat column-major array from the row-major data
    std::vector<double> X_flat = CSVLoader::transpose_flat(X);

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

    if (rank == 0)
        cout << "[Timing] RandomForest MPI fit() total time: "
                << chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count()
                << " ms\n";
    return std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
}

// Predict for one sample
int RandomForest::predict(const vector<double> &x) const {
    unordered_map<int, int> vote_count;
    // Collect votes from each tree
    for (const auto &t: trees) {
        if (t.hasFlat() || t.root) // sanity check that tree is trained, in either representation
        {
            int p = t.predict(x);
            vote_count[p]++;
        }
    } // Return label with most votes
    return ranges::max_element(vote_count.begin(), vote_count.end(),
                               [](const auto &a, const auto &b) { return a.second < b.second; })->first;
}

// Batch prediction
vector<int> RandomForest::predict_batch(const vector<vector<double> > &X) const {
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const size_t N = X.size();
    const size_t chunk_size = (N + n_ranks - 1) / n_ranks;
    const size_t start = rank * chunk_size;
    const size_t end = min(start + chunk_size, N);

    if (verbose)
        cout << "[Rank " << rank << "] Processing indices " << start << " to " << end - 1 << endl;

    const auto startT = chrono::high_resolution_clock::now();
    // Local predictions
    vector<int> local_predictions(end - start);

#pragma omp parallel for schedule(static) default(none) shared(X, local_predictions,start, end)
    for (size_t i = start; i < end; ++i) {
        local_predictions[i - start] = predict(X[i]);
    }

    // If only one rank, return local predictions directly
    if (n_ranks == 1) {
        const auto endT = chrono::high_resolution_clock::now();
        cout << "[Timing Rank " << rank << "] RandomForest predict_batch() total time: "
                << chrono::duration_cast<chrono::milliseconds>(endT - startT).count()
                << " ms" << endl;
        return local_predictions;
    }

    // Prepare counts and displacements
    vector<int> counts(n_ranks);
    vector<int> displacements(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const size_t s = static_cast<size_t>(r) * chunk_size;
        const size_t e = min(s + chunk_size, N);
        counts[r] = static_cast<int>(e - s);
        displacements[r] = static_cast<int>(s);
    }

    // Gather predictions to rank 0
    vector<int> predictions;
    if (rank == 0)
        predictions.resize(N);

    MPI_Gatherv(local_predictions.data(), static_cast<int>(local_predictions.size()), MPI_INT,
                predictions.data(), counts.data(), displacements.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    const auto endT = chrono::high_resolution_clock::now();
    if (rank == 0)
        cout << "[Timing Rank " << rank << "] RandomForest predict_batch() total time: "
                << chrono::duration_cast<chrono::milliseconds>(endT - startT).count()
                << " ms" << endl;

    return predictions; // valid only on rank 0
}

std::vector<int> RandomForestDistributed::predict_batch(const std::vector<std::vector<double> > &X,
                                                        const bool distributionStrat) const {
    if (distributionStrat) {
        /// Use standard RandomForest batch prediction (data distributed)
        return RandomForest::predict_batch(X);
    }
    // Alternative version where the data is fully replicated on each rank, but each rank only has a subset of trees
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
    vector local_votes(N * n_classes, 0);
#pragma omp parallel for schedule(static) default(none) shared(X, local_votes, start_tree, end_tree, trees, N, n_classes\
)
    for (size_t i = 0; i < N; ++i) {
        for (size_t t = start_tree; t < end_tree; ++t) {
            const int p = trees[t].predict(X[i]);
#pragma omp atomic
            local_votes[i * n_classes + p]++;
        }
    }
    // Reduce votes across all ranks
    vector global_votes(N * n_classes, 0);
    MPI_Allreduce(local_votes.data(), global_votes.data(), static_cast<int>(N * n_classes), MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    // Determine final predictions
    vector<int> predictions(N);
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
    if (rank == 0)
        cout << "[Timing Rank " << rank << "] RandomForest predict_batch() total time: "
                << chrono::duration_cast<chrono::milliseconds>(endT - startT).count()
                << " ms" << endl;
    return predictions;
}


void RandomForestDistributed::gather_all_trees(MPI_Comm comm) {
    int rank, n_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    const int total_trees = static_cast<int>(trees.size());

    // --------- MPI datatype for FlatNode ----------
    MPI_Datatype MPI_FlatNode;
    const int block_lengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint offsets[5];
    const MPI_Datatype types[5] = {MPI_C_BOOL, MPI_INT, MPI_DOUBLE, MPI_INT, MPI_INT};

    offsets[0] = offsetof(FlatNode, is_leaf);
    offsets[1] = offsetof(FlatNode, feature);
    offsets[2] = offsetof(FlatNode, threshold);
    offsets[3] = offsetof(FlatNode, label);
    offsets[4] = offsetof(FlatNode, right);

    MPI_Type_create_struct(5, block_lengths, offsets, types, &MPI_FlatNode);
    MPI_Type_commit(&MPI_FlatNode);

    // --------- Build local sizes + local buffer ----------
    std::vector<int> local_tree_sizes;
    local_tree_sizes.reserve(total_trees);
    std::vector<FlatNode> local_buffer;
    local_buffer.reserve(1024);

    for (int t = 0; t < total_trees; ++t) {
        if (DecisionTree &dt = trees[t]; dt.root) {
            auto flat = dt.getFlat();
            local_tree_sizes.push_back(static_cast<int>(flat.size()));
            local_buffer.insert(local_buffer.end(), flat.begin(), flat.end());
        } else {
            local_tree_sizes.push_back(0);
        }
    }

    // --------- Gather per-slot sizes (fixed-count) ----------
    std::vector<int> all_tree_sizes(n_ranks * total_trees);
    MPI_Allgather(local_tree_sizes.data(), total_trees, MPI_INT,
                  all_tree_sizes.data(), total_trees, MPI_INT,
                  comm);

    // --------- Gather node counts ----------
    const int local_nodes = static_cast<int>(local_buffer.size());
    std::vector<int> all_nodes_counts(n_ranks);
    MPI_Allgather(&local_nodes, 1, MPI_INT,
                  all_nodes_counts.data(), 1, MPI_INT,
                  comm);

    // compute displacements
    std::vector node_displacements(n_ranks, 0);
    for (int i = 1; i < n_ranks; ++i)
        node_displacements[i] = node_displacements[i - 1] + all_nodes_counts[i - 1];

    const int total_nodes =
            std::accumulate(all_nodes_counts.begin(), all_nodes_counts.end(), 0);

    // --------- Gather nodes ----------
    std::vector<FlatNode> all_nodes(total_nodes);
    MPI_Allgatherv(local_buffer.data(), local_nodes, MPI_FlatNode,
                   all_nodes.data(), all_nodes_counts.data(), node_displacements.data(), MPI_FlatNode,
                   comm);

    MPI_Type_free(&MPI_FlatNode);

    // --------- Reconstruct full forest ----------
    std::vector<std::vector<FlatNode> > full_forest;
    full_forest.resize(total_trees);
    std::vector<int> read_pos = node_displacements;

    for (int t = 0; t < total_trees; ++t) {
        bool filled = false;

        for (int r = 0; r < n_ranks; ++r) {
            if (const int sz = all_tree_sizes[r * total_trees + t]; sz > 0) {
                // ========================================================
                // LOG BLOCK #4 — reconstruction choice
                // ========================================================
                if (rank == 0 && verbose)
                    std::cerr << "Slot " << t << " reconstructed from rank "
                            << r << " (sz=" << sz << ")\n";

                const int start = read_pos[r];
                full_forest[t].assign(all_nodes.begin() + start,
                                      all_nodes.begin() + start + sz);
                read_pos[r] += sz;
                filled = true;
                break;
            }
        }

        if (!filled) {
            full_forest[t].clear();
        }
    }

    // --------- Load final trees ----------
    for (int t = 0; t < total_trees; ++t) {
        if (!full_forest[t].empty())
            trees[t].set_flat(std::move(full_forest[t]));
    }
}
