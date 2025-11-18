#include "DecisionTreeIndexed.hpp"
#include <vector>
#include <memory>
#include <random>
#include <omp.h>

// Train n_trees in parallel using OpenMP.
// X, y: full dataset
// n_trees: number of trees
// sample_size: number of samples per bootstrap (if == 0, use X.size())
// base_seed: base for per-tree deterministic seeding
std::vector<std::unique_ptr<DecisionTree>> train_forest_openmp(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    const int n_trees,
    int max_depth,
    int min_samples,
    size_t sample_size,
    const unsigned int base_seed)
{
    if (sample_size == 0) sample_size = X.size();
    std::vector<std::unique_ptr<DecisionTree>> forest(n_trees);

#pragma omp parallel for schedule(static)
    for (int t = 0; t < n_trees; ++t) {
        unsigned int tree_seed = base_seed + static_cast<unsigned int>(t);
        // bootstrap indices with replacement
        std::mt19937 rng(tree_seed);
        std::uniform_int_distribution<size_t> dist(0, X.size() - 1);
        std::vector<size_t> sample_idx;
        sample_idx.reserve(sample_size);
        for (size_t i = 0; i < sample_size; ++i) sample_idx.push_back(dist(rng));

        auto tree = std::make_unique<DecisionTree>(max_depth, min_samples, tree_seed);
        tree->fit(X, y, sample_idx);
        forest[t] = std::move(tree);
    }

    return forest;
}
