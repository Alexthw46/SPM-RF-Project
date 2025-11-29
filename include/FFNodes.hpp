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

// Lightweight task describing a single tree training job.
struct TreeTask {
    DecisionTree &tree;
    std::vector<size_t> &bootstrap_idx;
    const ColMajorViewFlat &X;
    const std::vector<int> &y;
};

struct TreeRangeTask {
    std::vector<DecisionTree>* trees;
    const ColMajorViewFlat* X;
    const std::vector<int>* y;
    size_t begin;
    size_t end;
    uint64_t seed_base;
};


// Task describing a chunk of samples to predict.
struct PredictTask {
    const RandomForest &forest;
    std::span<const std::vector<double>> X_chunk;
    std::vector<int> &predictions;
    size_t offset;
};

constexpr bool ff_verbose = false;

// Worker that trains a tree
class TreeWorker final : public ff::ff_node {
public:
    void *svc(void *task_ptr) override {
        if (task_ptr == FF_EOS) return FF_EOS;

        auto &[tree, bootstrap_idx, X, y] = *static_cast<TreeTask *>(task_ptr);

        const auto t_start = std::chrono::high_resolution_clock::now();
        tree.fit(X, y, bootstrap_idx);
        const auto t_end = std::chrono::high_resolution_clock::now();

        if (ff_verbose)
            std::cout << "[Timing] Tree trained in "
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
                      << " ns\n";

        return GO_ON;
    }
};

class TreeWorker2 final : public ff::ff_node {
public:
    void* svc(void* task_ptr) override {
        if (task_ptr == FF_EOS) return FF_EOS;

        const auto& task = *static_cast<TreeRangeTask*>(task_ptr);
        auto& trees = *task.trees;
        auto& X = *task.X;
        auto& y = *task.y;

        const size_t n = X.n_samples;

        for (size_t i = task.begin; i < task.end; ++i) {
            std::mt19937 rng(task.seed_base + i);
            std::uniform_int_distribution<size_t> dist(0, n - 1);

            // bootstrap built *inside the worker*
            std::vector<size_t> idx(n);
            for (size_t j = 0; j < n; ++j)
                idx[j] = dist(rng);

            trees[i].fit(X, y, idx);
        }

        return GO_ON;
    }
};


// Worker that predicts a chunk
class PredictWorker final : public ff::ff_node {
public:
    void *svc(void *task_ptr) override {
        if (task_ptr == FF_EOS) return FF_EOS;

        auto &[forest, X_chunk, predictions, offset] = *static_cast<PredictTask *>(task_ptr);
        const size_t N = X_chunk.size();

        for (size_t i = 0; i < N; ++i)
            predictions[offset + i] = forest.predict(X_chunk[i]);

        return GO_ON;
    }
};

// Emitter that sends tree build tasks
class TreeBuildEmitter final : public ff::ff_node {
public:
    const ColMajorViewFlat &X;
    const std::vector<int> &y;
    std::vector<DecisionTree> &trees;
    std::vector<std::vector<size_t> > &bootstrap_indices_per_tree;
    std::vector<TreeTask> tasks;

    TreeBuildEmitter(std::vector<DecisionTree> &t,
                     std::vector<std::vector<size_t> > &idx,
                     const ColMajorViewFlat &X_,
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

    void *svc(void *) override {
        for (auto &task: tasks)
            ff_send_out(&task);
        return FF_EOS;
    }
};

class TreeBuildEmitter2 final : public ff::ff_node {
public:
    std::vector<TreeRangeTask> tasks;

    TreeBuildEmitter2(std::vector<DecisionTree>& trees,
                     const ColMajorViewFlat& X,
                     const std::vector<int>& y,
                     const size_t nWorkers,
                     const uint64_t seed_base)
    {
        const size_t n = trees.size();
        const size_t block = (n + nWorkers - 1) / nWorkers;

        tasks.reserve(nWorkers);

        for (size_t w = 0; w < nWorkers; ++w) {
            const size_t begin = w * block;
            const size_t end   = std::min(begin + block, n);
            if (begin >= end) break;

            tasks.push_back(TreeRangeTask{
                &trees,
                &X,
                &y,
                begin,
                end,
                seed_base
            });
        }
    }

    void* svc(void*) override {
        for (auto& t : tasks)
            ff_send_out(&t);
        return FF_EOS;
    }
};

// Emitter that sends prediction tasks
class PredictEmitter final : public ff::ff_node {
public:
    const RandomForest &forest;
    const std::vector<std::vector<double> > &X;
    std::vector<int> &predictions;
    std::vector<PredictTask> tasks;

    PredictEmitter(const RandomForest &f,
                   const std::vector<std::vector<double> > &X_,
                   std::vector<int> &preds,
                   const size_t chunk_size = 1000)
        : forest(f), X(X_), predictions(preds) {
        const size_t N = X.size();
        size_t offset = 0;

        while (offset < N) {
            const size_t end = std::min(offset + chunk_size, N);
            tasks.emplace_back(PredictTask{
                forest,
                std::span(X.data() + offset, end - offset),
                predictions,
                offset
            });

            offset = end;
        }
    }

    void *svc(void *) override {
        for (auto &task: tasks)
            ff_send_out(&task);
        return FF_EOS;
    }
};
