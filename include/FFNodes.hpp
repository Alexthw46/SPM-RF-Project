#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <span>
#include <vector>
// ReSharper disable once CppUnusedIncludeDirective
#include <ff/ff.hpp>
#include <ff/farm.hpp>

#include "DecisionTreeIndexed.hpp"
#include "RandomForestIndexed.hpp"

using namespace ff;
using namespace std;

/**
* @brief Task describing work to train a single DecisionTree.
*
* @param tree Reference to the DecisionTree to be trained.
* @param X Reference to the feature matrix in column-major format.
* @param y Reference to the vector of target labels.
* @param task_seed Seed for bootstrap sampling.
*/
struct TreeTask {
    DecisionTree &tree;
    const ColMajorViewFlat &X;
    const vector<int> &y;
    uint64_t task_seed;
};

/**
 * @brief Task describing work to train a range of DecisionTrees.
 *
 * @param trees Reference to the vector of DecisionTrees to be trained.
 * @param X Reference to the feature matrix in column-major format.
 * @param y Reference to the vector of target labels.
 * @param begin Starting index of the tree range (inclusive).
 * @param end Ending index of the tree range (exclusive).
 * @param seed_base Base seed for bootstrap sampling, offset by tree index.
 */
struct TreeRangeTask {
    vector<DecisionTree> &trees;
    const ColMajorViewFlat &X;
    const vector<int> &y;
    size_t begin;
    size_t end;
    uint64_t seed_base;
};

/**
 * @brief Task describing work to predict a chunk of samples.
 *
 * @param forest Reference to the RandomForest used for predictions.
 * @param X_chunk Span of feature vectors to be predicted.
 * @param predictions Reference to the vector where predictions will be stored.
 * @param offset Offset in the predictions vector corresponding to the start of X_chunk.
 */
struct PredictTask {
    const RandomForest &forest;
    span<const vector<double>> X_chunk;
    vector<int> &predictions;
    size_t offset;
};

constexpr bool ff_verbose = false;

/**
 * @brief FastFlow Worker that trains a single DecisionTree.
 *
 * @note Each worker maintains its own RNG and bootstrap indices buffer to reuse the same memory slice across multiple trees.
 */
class TreeWorker final : public ff_node {
public:
    // Initialize the worker thread with explicit CPU pinning
    int svc_init() override {
        ff_mapThreadToCpu(static_cast<int>(get_my_id())); // pins worker i to core i
        return 0;
    }

    /**
     * @brief Process a single \c TreeTask.
     *
     * Expected behavior:
     *  - If \p task_ptr is \c FF_EOS, propagate end-of-stream by returning \c FF_EOS.
     *  - Otherwise, interpret \p task_ptr as a \c TreeTask reference.
     *  - Seed the per-worker RNG with the provided seed and build a uniform
     *    distribution over sample indices.
     *  - Call \c DecisionTree::fit with the bootstrap samples drawn from the distribution.
     *  - If ff_verbose is enabled, print the time taken to train the tree.
     *
     * @param task_ptr Pointer to a \c TreeTask or the \c FF_EOS sentinel.
     * @return \c GO_ON to continue processing, or \c FF_EOS to stop.
     */
    void *svc(void *task_ptr) override {
        if (task_ptr == FF_EOS) return FF_EOS;

        auto &[tree, X, y, seed] = *static_cast<TreeTask *>(task_ptr);

        const auto t_start = chrono::high_resolution_clock::now();
        // Bootstrap sampling
        rng.seed(seed);
        dist = uniform_int_distribution<size_t>(0, X.n_samples - 1);

        bootstrap_indices.resize(X.n_samples);

        // Bootstrap indices
        for (size_t j = 0; j < X.n_samples; ++j)
            bootstrap_indices[j] = dist(rng);

        tree.fit(X, y, bootstrap_indices);
        const auto t_end = chrono::high_resolution_clock::now();

        if (ff_verbose)
            cout << "[Timing] Tree trained in "
                    << chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count()
                    << " ns\n";

        return GO_ON;
    }

    // allocated once per worker
private:
    /** @brief Per-worker Mersenne Twister RNG used for bootstrap sampling. */
    mt19937 rng;
    /** @brief Uniform distribution used to draw bootstrap sample indices. */
    uniform_int_distribution<size_t> dist;
    /** @brief Reusable buffer holding bootstrap indices for the current fit call. */
    vector<size_t> bootstrap_indices;
};

/**
 * @brief FastFlow Worker node that predicts a contiguous chunk of samples.
 *
 *
 */
class PredictWorker final : public ff_node {
    /**
     * @brief Initialize the worker thread with explicit CPU pinning.
     */
    int svc_init() override {
        ff_mapThreadToCpu(static_cast<int>(get_my_id())); // pins worker i to core i
        return 0;
    }

public:
    /**
     * @brief Process a single `PredictTask`.
     *
     * Expected behavior:
     *  - If `task_ptr` == `FF_EOS`, propagate end-of-stream by returning `FF_EOS`.
     *  - Otherwise, cast `task_ptr` to `PredictTask *` and destructure the task.
     *  - For each sample in the chunk, call `RandomForest::predict` and store the
     *    resulting label into `predictions[offset + i]`.
     *
     * @param task_ptr Pointer to a `PredictTask` or the `FF_EOS` sentinel.
     * @return void* `GO_ON` to continue processing, or `FF_EOS` to stop.
     */
    void *svc(void *task_ptr) override {
        if (task_ptr == FF_EOS) return FF_EOS;

        auto &[forest, X_chunk, predictions, offset] = *static_cast<PredictTask *>(task_ptr);
        const size_t N = X_chunk.size();

        for (size_t i = 0; i < N; ++i)
            predictions[offset + i] = forest.predict(X_chunk[i]);

        return GO_ON;
    }
};

/**
 * @class TreeBuildEmitter
 * @brief FastFlow emitter node that creates and emits TreeTask objects for each
 *        DecisionTree of the forest.
 */
class TreeBuildEmitter final : public ff_node {
public:
    /** @brief Reference to the feature matrix in column-major format. */
    const ColMajorViewFlat &X;
    /** @brief Reference to the vector of target labels. */
    const vector<int> &y;
    /** @brief Reference to the vector of DecisionTree to be trained. */
    vector<DecisionTree> &trees;
    /** @brief Pre-built tasks that will be emitted to workers. */
    vector<TreeTask> tasks;
    /** @brief Preseeded RNG used to produce bootstrap seeds for each task. */
    mt19937 gen;

    /**
     * @brief Construct a TreeBuildEmitter.
     *
     * This constructor fills the internal \c tasks vector with one \c TreeTask
     * per tree in \p t. Each task stores references to \p X and \p y and a seed
     * produced by \p gen.
     *
     * @param t Reference to the vector of DecisionTree objects to build tasks for.
     * @param X_ Reference to the feature matrix view.
     * @param y_ Reference to the target labels vector.
     * @param gen Reference to a random engine used to produce seeds for bootstrap sampling.
     */
    TreeBuildEmitter(vector<DecisionTree> &t,
                     const ColMajorViewFlat &X_,
                     const vector<int> &y_,
                     mt19937 &gen
    )
        : X(X_), y(y_), trees(t), gen(gen) {
        tasks.reserve(trees.size());
        for (DecisionTree &tree: trees) {
            tasks.emplace_back(TreeTask{
                tree,
                X,
                y,
                gen()
            });
        }
    }

    /**
     * @brief Emit all prepared TreeTask to the FastFlow workers.
     *
     * @return \c FF_EOS sentinel after emitting all tasks.
     */
    void *svc(void *) override {
        for (auto &task: tasks)
            ff_send_out(&task);
        return FF_EOS;
    }
};

/**
 * @class PredictEmitter
 * @brief FastFlow emitter node that creates and emits PredictTask objects for chunks of samples.
 */
class PredictEmitter final : public ff_node {
public:
    /** @brief Reference to the RandomForest used for predictions. */
    const RandomForest &forest;
    /** @brief Reference to the feature matrix in row-major format. */
    const vector<vector<double> > &X;
    /** @brief Reference to the vector where predictions will be stored. */
    vector<int> &predictions;
    /** @brief Pre-built tasks that will be emitted to workers. */
    vector<PredictTask> tasks;

    /**
     * @brief Construct a PredictEmitter.
     *
     * This constructor divides the input feature matrix \p X into chunks of size
     * \p chunk_size and creates a \c PredictTask for each chunk, storing them in
     * the internal \c tasks vector.
     *
     * @param f Reference to the RandomForest used for predictions.
     * @param X_ Reference to the feature matrix in row-major format.
     * @param preds Reference to the vector where predictions will be stored.
     * @param chunk_size Size of each chunk of samples to predict.
     */
    PredictEmitter(const RandomForest &f,
                   const vector<vector<double> > &X_,
                   vector<int> &preds,
                   const size_t chunk_size = 1000)
        : forest(f), X(X_), predictions(preds) {
        const size_t N = X.size();
        size_t offset = 0;

        while (offset < N) {
            const size_t end = min(offset + chunk_size, N);
            tasks.emplace_back(PredictTask{
                forest,
                span(X.data() + offset, end - offset),
                predictions,
                offset
            });

            offset = end;
        }
    }

    /**
     * @brief Emit all prepared PredictTask to the FastFlow workers.
     *
     * @return \c FF_EOS sentinel after emitting all tasks.
     */
    void *svc(void *) override {
        for (auto &task: tasks)
            ff_send_out(&task);
        return FF_EOS;
    }
};

/**
 * @brief FastFlow Worker that trains a range of DecisionTrees.
 *
 * @note Each worker maintains its own RNG and bootstrap indices buffer, to reuse the same memory slice across multiple trees.
 *
 *
 **/
class TreeWorkerRange final : public ff_node {
public:
    // Initialize the worker thread with explicit CPU pinning
    int svc_init() override {
        ff_mapThreadToCpu(static_cast<int>(get_my_id())); // pins worker i to core i
        return 0;
    }

    /**
     * @brief Process a single \c TreeRangeTask.
     *
     * Expected behavior:
     *  - If \p task_ptr is \c FF_EOS, propagate end-of-stream by returning \c FF_EOS.
     *  - Otherwise, interpret \p task_ptr as a \c TreeRangeTask reference
     *  - For each tree in the specified range:
     *     - Seed the per-worker RNG with \c seed_base + tree index and bootstrap sample indices.
     *     - Call \c DecisionTree::fit with the bootstrap samples drawn from the distribution.
     *     - If ff_verbose is enabled, print the time taken to train the tree.
     *
     * @param task_ptr Pointer to a \c TreeRangeTask or the \c FF_EOS sentinel.
     * @return \c GO_ON to continue processing, or \c FF_EOS to stop.
     */
    void *svc(void *task_ptr) override {
        if (task_ptr == FF_EOS) return FF_EOS;

        auto &[trees, X, y, begin, end, seed_base] = *static_cast<TreeRangeTask *>(task_ptr);

        const size_t n = X.n_samples;

        bootstrap_indices.resize(n); // reuse buffer across trees

        for (size_t i = begin; i < end; ++i) {
            const auto t_start = chrono::high_resolution_clock::now();

            rng.seed(seed_base + i);
            dist = uniform_int_distribution<size_t>(0, n - 1);

            // Fill bootstrap indices
            for (size_t j = 0; j < n; ++j)
                bootstrap_indices[j] = dist(rng);

            trees[i].fit(X, y, bootstrap_indices);
            const auto t_end = chrono::high_resolution_clock::now();

            if (ff_verbose)
                cout << "[Timing] Tree trained in "
                        << chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count()
                        << " ns\n";
        }

        return GO_ON;
    }

    // allocated once per worker
private:
    /** @brief Per-worker Mersenne Twister RNG used for bootstrap sampling. */
    mt19937 rng;
    /** @brief Uniform distribution used to draw bootstrap sample indices. */
    uniform_int_distribution<size_t> dist;
    /** @brief Reusable buffer holding bootstrap indices for the current fit call. */
    vector<size_t> bootstrap_indices;
};

/**
 * @brief FastFlow emitter that creates and emits TreeRangeTask objects.
 *
 */
class TreeBuildEmitterRange final : public ff_node {
public:
    /**
     * @brief Prebuilt tasks that will be emitted to workers.
     *
     * Each element is a TreeRangeTask describing a contiguous range of trees
     * to be trained by a single worker.
     */
    vector<TreeRangeTask> tasks;

    /**
     * @brief Construct the emitter and partition work among workers.
     *
     * The number of trees is divided into \p nWorkers blocks.
     * For each non-empty block a TreeRangeTask is created with the
     * corresponding begin-end indices and the provided \p seed_base.
     *
     * @param trees Reference to the vector of DecisionTree objects to train.
     * @param X Reference to the column-major feature matrix view.
     * @param y Reference to the vector of target labels.
     * @param nWorkers Number of worker partitions to produce (at most).
     * @param seed_base Base seed used by workers to derive per-tree seeds.
     */
    TreeBuildEmitterRange(vector<DecisionTree> &trees,
                          const ColMajorViewFlat &X,
                          const vector<int> &y,
                          const size_t nWorkers,
                          const uint64_t seed_base) {
        // Compute block size
        const size_t n = trees.size();
        const size_t block = (n + nWorkers - 1) / nWorkers;

        // Preallocate tasks
        tasks.reserve(nWorkers);

        for (size_t w = 0; w < nWorkers; ++w) {
            // Compute range for this worker
            const size_t begin = w * block;
            const size_t end = min(begin + block, n);
            if (begin >= end) break;
            // Create task for this range of trees
            tasks.push_back(TreeRangeTask{
                trees,
                X,
                y,
                begin,
                end,
                seed_base
            });
        }
    }

    /**
     * @brief Emit all prepared TreeRangeTask objects to FastFlow workers.
     *
     * @return void* \c FF_EOS to signal end-of-stream.
     */
    void *svc(void *) override {
        for (auto &t: tasks)
            ff_send_out(&t);
        return FF_EOS;
    }
};