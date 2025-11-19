#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <unordered_map>

// ReSharper disable once CppUnusedIncludeDirective
#include <ff/ff.hpp>
#include <ff/farm.hpp>

#include "DecisionTreeIndexed.hpp"
#include "RandomForestIndexed.hpp"

using namespace std;
using namespace ff;

/**
 * @brief Lightweight task describing a single tree training job.
 *
 * TreeTask aggregates non-owning references required by a worker to train
 * one DecisionTree:
 *  - a reference to the DecisionTree instance to train,
 *  - a reference to the precomputed bootstrap index vector for that tree,
 *  - a const reference to the feature matrix \p X,
 *  - a const reference to the label vector \p y.
 *
 * Ownership and lifetime:
 *  - TreeTask does not own any referenced object. Callers must ensure the
 *    referenced DecisionTree, bootstrap index vector, feature matrix and
 *    label vector stay alive until the worker has processed the task.
 *  - Addresses of TreeTask objects may be sent to worker threads; therefore
 *    the TreeTask instance itself must also remain valid until the worker
 *    finishes using it.
 */
struct TreeTask {
    /// DecisionTree to train (non-owning reference).
    DecisionTree &tree;

    /// Bootstrap sample indices for this tree (non-owning reference).
    std::vector<size_t> &bootstrap_idx;

    /// Read-only feature matrix (rows = samples).
    const std::vector<std::vector<double> > &X;

    /// Read-only label vector (one label per sample).
    const std::vector<int> &y;
};


/**
 * @brief Task describing a chunk of samples to predict.
 *
 * This lightweight POD aggregates non-owning references required by a worker
 * to perform predictions for a contiguous slice of the input matrix:
 *
 * - \p forest: const reference to the RandomForest used for per-sample prediction.
 * - \p X_chunk: zero-copy slice (std::span) referencing a contiguous range of
 *   rows from the feature matrix (each row is a std::vector<double>).
 * - \p predictions: reference to the output vector where predicted labels
 *   for this chunk will be stored.
 * - \p offset: index in \p predictions that corresponds to the first element
 *   of \p X_chunk.
 *
 * Ownership and lifetime:
 * - This struct does not own any referenced objects. The caller/emitter must
 *   ensure that the RandomForest instance, the backing container for the
 *   span and the predictions vector outlive the worker processing this task.
 *
 * Thread-safety:
 * - Workers may write to distinct, non-overlapping ranges of \p predictions.
 *   The emitter must ensure tasks are created with non-overlapping offsets.
 */
struct PredictTask {
    const RandomForest &forest;
    std::span<const std::vector<double>> X_chunk; // zero-copy slice
    std::vector<int> &predictions;
    size_t offset;
};


constexpr bool verbose = false;

/**
 * @class TreeWorker
 * @brief FastFlow worker node that trains a DecisionTree from a TreeTask.
 *
 * Behavior:
 *  - Receives a pointer which is either a pointer to a TreeTask or the
 *    FastFlow sentinel \c FF_EOS.
 *  - If \c FF_EOS is received the worker forwards it and returns \c FF_EOS.
 *  - Otherwise the pointer is cast to \c TreeTask*, data are unpacked and
 *    \c DecisionTree::fit is called with the provided bootstrap indices.
 *
 * Threading and safety:
 *  - This node does not create or own the referenced objects; the producer
 *    must guarantee their lifetime for the duration of the task processing.
 *  - Logging is conditional on the module-level \c verbose flag.
 */
class TreeWorker final : public ff_node {
public:
    /**
     * FastFlow service method invoked for each incoming message.
     *
     * @param task_ptr Pointer to a \c TreeTask or \c FF_EOS sentinel.
     * @return \c FF_EOS when end-of-stream is received; otherwise \c GO_ON.
     */
    void *svc(void *task_ptr) override {
        if (task_ptr == FF_EOS) return FF_EOS;

        auto &[tree, bootstrap_idx, X, y] = *static_cast<TreeTask *>(task_ptr);

        const auto t_start = std::chrono::high_resolution_clock::now();
        tree.fit(X, y, bootstrap_idx);
        const auto t_end = std::chrono::high_resolution_clock::now();

        if (verbose)
            std::cout << "[Timing] Tree trained in "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
                    << " ns\n";

        return GO_ON;
    }
};

/**
 * @class PredictWorker
 * @brief FastFlow worker node that performs predictions for a chunk of samples.
 *
 * Behavior:
 *  - Receives a pointer which is either a pointer to a \c PredictTask or the
 *    FastFlow sentinel \c FF_EOS.
 *  - If \c FF_EOS is received the worker forwards it and returns \c FF_EOS.
 *  - Otherwise the pointer is cast to \c PredictTask*, the task is unpacked and
 *    per-sample predictions are computed by calling \c RandomForest::predict.
 *
 * Thread-safety:
 *  - Tasks must ensure non-overlapping write ranges into the shared
 *    \c predictions vector. Each worker writes only to the range specified by
 *    the task's \c offset and the chunk length.
 */
class PredictWorker final : public ff_node {
public:
    /**
     * FastFlow service method invoked for each incoming message.
     *
     * @param task_ptr Pointer to a \c PredictTask or \c FF_EOS sentinel.
     * @return \c FF_EOS when end-of-stream is received; otherwise \c GO_ON.
     *
     * Notes:
     *  - The PredictTask contains non-owning references; the caller must ensure
     *    the referenced RandomForest instance, the backing container for the
     *    span and the predictions vector outlive this call.
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
 * @brief Emitter node that prepares and sends TreeTask messages to workers.
 *
 * Responsibilities:
 *  - Construct a local vector of \c TreeTask entries (one per tree).
 *  - Emit the address of each \c TreeTask to the FastFlow workers via \c ff_send_out.
 *
 * Important lifetime and ownership notes:
 *  - The emitter stores references to external containers: \c trees,
 *    \c bootstrap_indices_per_tree, \c X and \c y. Those containers MUST outlive the
 *    emitter and the entire FastFlow farm run; otherwise the references inside
 *    each \c TreeTask will become dangling.
 *  - The emitter's \c tasks vector owns the \c TreeTask objects and the emitter
 *    sends the address of each element (\c &task) to workers. The application
 *    must ensure the \c tasks vector remains valid until workers have consumed
 *    and processed the pointers (FastFlow does not copy the pointed-to task).
 *
 * Threading:
 *  - Emission occurs in the emitter thread only; workers run concurrently.
 *  - This class does not transfer ownership of the referenced containers.
 */
class TreeBuildEmitter final : public ff_node {
public:
    /// Read-only reference to feature matrix.
    const std::vector<std::vector<double> > &X;
    /// Read-only reference to labels.
    const std::vector<int> &y;
    /// Reference to the vector of DecisionTree instances to be trained.
    std::vector<DecisionTree> &trees;
    /// Reference to the precomputed bootstrap indices for each tree.
    std::vector<std::vector<size_t> > &bootstrap_indices_per_tree;
    /// Local storage of tasks; addresses of these elements are emitted to workers.
    std::vector<TreeTask> tasks;

    /**
     * Construct an emitter and prepare one TreeTask per tree.
     *
     * @param t      Reference to vector of DecisionTree instances (must outlive emitter).
     * @param idx    Reference to vector of bootstrap index vectors (must outlive emitter).
     * @param X_     Reference to feature matrix (read-only, must outlive emitter).
     * @param y_     Reference to label vector (read-only, must outlive emitter).
     *
     * The constructor fills \c tasks so each element holds non-owning references
     * into the provided containers. No dataset copying occurs.
     */
    TreeBuildEmitter(std::vector<DecisionTree> &t,
                     std::vector<std::vector<size_t> > &idx,
                     const std::vector<std::vector<double> > &X_,
                     const std::vector<int> &y_)
        : X(X_), y(y_), trees(t), bootstrap_indices_per_tree(idx) {
        tasks.reserve(trees.size());
        for (size_t i = 0; i < trees.size(); i++) {
            tasks.emplace_back(TreeTask{
                trees[i],
                bootstrap_indices_per_tree[i],
                X,
                y
            });
        }
    }

    /**
     * Emit all prepared tasks to the FastFlow workers.
     *
     * After sending all tasks this emitter returns \c FF_EOS to indicate
     * end-of-stream.
     *
     * @return \c FF_EOS pointer to signal no more tasks.
     */
    void *svc(void *) override {
        for (auto &task: tasks)
            ff_send_out(&task);
        return FF_EOS;
    }
};

/**
 * @class PredictEmitter
 * @brief FastFlow emitter that prepares and sends prediction tasks.
 *
 * The emitter partitions the input feature matrix into contiguous chunks
 * and builds a vector of non-owning \c PredictTask entries. Each task
 * holds a \c std::span referencing a contiguous range of rows from \c X
 * and the offset in the shared \c predictions output vector where results
 * for that chunk must be written.
 *
 * Lifetime and ownership:
 * - All references stored in this class are non-owning. The caller must
 *   ensure that the referenced \c RandomForest instance, the backing
 *   storage for \c X and the \c predictions vector
 *   outlive the emitter and the farm run.
 *
 * Thread-safety:
 * - Workers must write to non-overlapping ranges of \c predictions as
 *   specified by each task's \c offset and chunk length.
 */
class PredictEmitter final : public ff_node {
public:
    /// Non-owning reference to the RandomForest used for per-sample prediction.
    const RandomForest &forest;
    /// Non-owning reference to the feature matrix (rows = samples).
    const std::vector<std::vector<double> > &X;
    /// Reference to the shared output vector where predictions will be written.
    std::vector<int> &predictions;
    /// Local storage of prepared tasks. Addresses of these elements are emitted.
    std::vector<PredictTask> tasks;

    /**
     * @brief Construct an emitter that partitions \c X into chunks.
     *
     * @param f        Non-owning reference to the RandomForest used for predictions.
     * @param X_       Non-owning reference to the feature matrix (rows = samples).
     * @param preds    Reference to the output predictions vector where workers will write labels.
     * @param chunk_size Maximum number of samples per emitted task (default: 1000).
     *
     * The constructor builds \c tasks so each PredictTask references a contiguous
     * slice of the backing container for \c X using \c std::span. No rows are copied.
     * The caller must ensure that the backing storage of \c X and the \c predictions
     * vector remain valid while the farm processes the emitted task pointers.
     */
    PredictEmitter(const RandomForest &f,
                   const std::vector<std::vector<double> > &X_,
                   std::vector<int> &preds,
                   const size_t chunk_size = 1000)
        : forest(f), X(X_), predictions(preds) {
        const size_t N = X.size();
        size_t offset = 0;

        while (offset < N) {
            const size_t end = std::min(offset + chunk_size, N);
            // Create a task referencing the slice
            tasks.emplace_back(PredictTask{
                forest,
                std::span(X.data() + offset, end - offset),
                predictions,
                offset
            });

            offset = end;
        }
    }

    /**
     * @brief Emit all prepared \c PredictTask pointers to FastFlow workers.
     *
     * Sends the address of each element in \c tasks via \c ff_send_out.
     * After all tasks are sent this emitter returns \c FF_EOS to signal
     * end-of-stream to downstream workers.
     *
     * @return \c FF_EOS pointer to indicate end-of-stream.
     */
    void *svc(void *) override {
        for (auto &task: tasks)
            ff_send_out(&task);
        return FF_EOS;
    }
};

// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), gen(seed) {
    int min_samples = 2;
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, min_samples, seed + i); // each tree gets unique deterministic seed
}

/**
 * Create bootstrap sample indices for a single tree.
 *
 * Fills `bootstrap_idx` with `X.size()` indices sampled uniformly with replacement
 * from the range [0, X.size() - 1] using the provided `dist` and the supplied RNG.
 *
 * @param X            The dataset (used only to determine the number of samples).
 * @param dist         Uniform integer distribution to draw indices from (passed by const reference).
 * @param rng          Pseudo-random number generator to draw samples from (stateful; use a seeded rng for deterministic results).
 * @param bootstrap_idx Output vector which will be resized to X.size() and
 *                      populated with sampled indices (indices into `X`).
 *
 * Notes:
 * - This function does not modify `X` or `dist`.
 * - Determinism: if `rng` is seeded deterministically prior to the call, the
 *   generated bootstrap indices will be deterministic.
 */
inline void create_bootstrap_indexes(
    const std::vector<std::vector<double> > &X,
    std::uniform_int_distribution<size_t> &dist,
    std::mt19937 &rng,
    std::vector<size_t> &bootstrap_idx) {
    const size_t n = X.size();
    bootstrap_idx.resize(n);

    for (size_t i = 0; i < n; i++)
        bootstrap_idx[i] = dist(rng);
}

// Train forest with bootstrap sampling (index-based, no copies)
// FastFlow version
void RandomForest::fit(const vector<vector<double> > &X, const vector<int> &y) {
    // Distribution to use for bootstrap sampling
    uniform_int_distribution<size_t> dist(0, X.size() - 1);
    // Precompute bootstrap indexes for all trees
    std::vector<std::vector<size_t> > bootstrap_indices_per_tree(trees.size());
    for (auto &vec: bootstrap_indices_per_tree)
        create_bootstrap_indexes(X, dist, gen, vec);

    vector<ff_node *> workers(ff_numCores());
    ranges::generate(workers, [] { return new TreeWorker(); });

    ff_farm farm(workers);
    farm.add_emitter(new TreeBuildEmitter(trees, bootstrap_indices_per_tree, X, y));
    const auto total_start = chrono::high_resolution_clock::now();
    farm.run_and_wait_end();
    const auto total_end = chrono::high_resolution_clock::now();
    cout << "All trees built in parallel." << endl;

    cout << "[Timing] RandomForest fit() total time: "
            << chrono::duration_cast<chrono::nanoseconds>(total_end - total_start).count()
            << " ns" << endl;
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

/**
 * @brief Perform batch prediction for a set of samples in parallel.
 *
 * This method partitions the input feature matrix \p X into contiguous chunks
 * and uses a FastFlow farm of \c PredictWorker nodes to compute labels for
 * each sample in parallel. An emitter (\c PredictEmitter) prepares
 * non-owning \c PredictTask entries that reference slices of \p X and the
 * shared \p predictions output vector. The farm is executed and the function
 * waits for all worker nodes to complete before returning results.
 *
 * @param X Input feature matrix where each row is a sample (read-only).
 * @return Vector of predicted integer labels, one per row in \p X.
 *
 * Notes and guarantees:
 * - The returned vector has size \c X.size() and contains the predicted label
 *   for each corresponding row of \p X.
 * - The emitter and workers hold non-owning references into \p X and the
 *   \p predictions vector; those containers must remain valid for the
 *   lifetime of the farm run.
 * - The function prints elapsed wall-clock time (nanoseconds) for the
 *   parallel prediction run to stdout.
 */
std::vector<int> RandomForest::predict_batch(const std::vector<std::vector<double> > &X) const {
    std::vector<int> predictions(X.size());
    const size_t nCores = ff_numCores();

    // Workers
    std::vector<ff_node *> workers(nCores);
    ranges::generate(workers, [] { return new PredictWorker(); });

    ff_farm farm(workers);
    farm.add_emitter(new PredictEmitter(*this, X, predictions, 50));

    const auto start = std::chrono::high_resolution_clock::now();
    farm.run_and_wait_end();
    const auto end = std::chrono::high_resolution_clock::now();

    std::cout << "[Timing] RandomForest predict_batch_parallel() total time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
            << " ns" << std::endl;

    return predictions;
}
