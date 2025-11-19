#include "RandomForestIndexed.hpp"
#include "DecisionTreeIndexed.hpp"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <ff/ff.hpp>
#include <ff/farm.hpp>
using namespace std;
using namespace ff;

struct TreeTask {
    DecisionTree *tree; // pointer to the specific tree
    std::vector<size_t> *bootstrap_idx; // indices for the bootstrap sample
    const std::vector<std::vector<double> > *X; // dataset (read-only)
    const std::vector<int> *y; // target labels (read-only)
};

constexpr bool verbose = false;

class TreeWorker final : public ff_node {
public:
    void *svc(void *task_ptr) override {
        if (task_ptr == FF_EOS) return FF_EOS;

        const auto task = static_cast<TreeTask *>(task_ptr);

        const auto t_start = std::chrono::high_resolution_clock::now();
        task->tree->fit(*task->X, *task->y, *task->bootstrap_idx);
        const auto t_end = std::chrono::high_resolution_clock::now();

        if (verbose)
            std::cout << "[Timing] Tree trained in "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
                    << " ns" << std::endl;

        return GO_ON;
    }
};


class TreeBuildEmitter final : public ff_node {
public:
    const std::vector<std::vector<double> > *X;
    const std::vector<int> *y;
    std::vector<DecisionTree> *trees;
    std::vector<std::vector<size_t> > *bootstrap_idx_all;
    std::vector<TreeTask> tasks; // preallocated

    TreeBuildEmitter(std::vector<DecisionTree> *t,
            std::vector<std::vector<size_t> > *idx,
            const std::vector<std::vector<double> > *X_,
            const std::vector<int> *y_)
        : X(X_), y(y_), trees(t), bootstrap_idx_all(idx) {
        // Preallocate all tasks
        tasks.reserve(trees->size());
        for (size_t i = 0; i < trees->size(); i++) {
            tasks.push_back(TreeTask{
                &(*trees)[i],
                &(*bootstrap_idx_all)[i],
                X,
                y
            });
        }
    }

    void *svc(void *) override {
        for (auto &task: tasks) {
            ff_send_out(&task); // send pointer to preallocated task
        }
        return FF_EOS;
    }
};


// Constructor
RandomForest::RandomForest(const int n_t, int max_depth, const int n_classes, const unsigned int seed)
    : n_trees(n_t), max_depth(max_depth), n_classes(n_classes), gen(seed) {
    // Initialize trees
    for (int i = 0; i < n_trees; i++)
        trees.emplace_back(max_depth, 2, seed + i); // each tree gets unique deterministic seed
}

void RandomForest::create_bootstrap_indexes(const vector<vector<double> > &X, uniform_int_distribution<size_t> dist,
                                            vector<size_t> &bootstrap_idx) {
    bootstrap_idx.reserve(X.size());
    for (size_t j = 0; j < X.size(); j++)
        bootstrap_idx.push_back(dist(gen));
}

// Train forest with bootstrap sampling (index-based, no copies)
void RandomForest::fit(const vector<vector<double> > &X, const vector<int> &y) {
    // Distribution to use for bootstrap sampling
    const uniform_int_distribution<size_t> dist(0, X.size() - 1);

    vector<vector<size_t> > bootstrap_idx_all;
    bootstrap_idx_all.resize(trees.size());


    // Create bootstrap samples for each tree
    for (size_t i = 0; i < trees.size(); i++) {
        create_bootstrap_indexes(X, dist, bootstrap_idx_all[i]);
    }

    std::vector<ff_node *> workers;
    for (int i = 0; i < ff_numCores(); i++)
        workers.push_back(new TreeWorker());

    ff_farm farm(workers);
    farm.add_emitter(new TreeBuildEmitter(&trees, &bootstrap_idx_all, &X, &y));
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

// Batch prediction
vector<int> RandomForest::predict_batch(const vector<vector<double> > &X) const {
    const auto start = chrono::high_resolution_clock::now();
    vector<int> predictions;
    predictions.reserve(X.size());
    for (auto &row: X)
        predictions.push_back(predict(row));
    const auto end = chrono::high_resolution_clock::now();
    cout << "[Timing] RandomForest predict_batch() total time: "
            << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
            << " ns" << endl;
    return predictions;
}
