#include "DecisionTreeO.hpp"
#include <ff/ff.hpp>
#include <ff/farm.hpp>
#include <vector>
#include <thread>
#include <memory>
#include <random>
#include <queue>
#include <mutex>

using namespace ff;

// A small task struct
struct BuildTask {
    int tree_id{};
    unsigned int seed{};
    std::vector<size_t> sample_idx;
};

// Collector will receive built trees (wrapped as raw pointer)
struct Result {
    int tree_id;
    DecisionTree* tree_ptr; // ownership transferred to collector thread
};

// Emitter node: creates tasks
class Emitter final : public ff_node {
public:
    Emitter(std::queue<BuildTask>& tasks, std::mutex& m) : tasks(tasks), mtx(m) {}
    void* svc(void*) override {
        std::lock_guard<std::mutex> lk(mtx);
        // feed tasks one by one to farm
        while (!tasks.empty()) {
            const BuildTask t = std::move(tasks.front());
            tasks.pop();
            ff_send_out(new BuildTask(t));
        }
        return EOS; // no more tasks
    }
private:
    std::queue<BuildTask>& tasks;
    std::mutex& mtx;
};

// Worker: builds a tree and returns Result*
class Worker final : public ff_node {
public:
    Worker(const std::vector<std::vector<double>>& X,
           const std::vector<int>& y,
           const int max_depth, const int min_samples)
        : X(X), y(y), max_depth(max_depth), min_samples(min_samples) {}

    void* svc(void* task) override {
        const auto* bt = static_cast<BuildTask*>(task);
        auto* tree = new DecisionTree(max_depth, min_samples, bt->seed);
        tree->fit(X, y, bt->sample_idx);
        auto* res = new Result{bt->tree_id, tree};
        delete bt;
        return res;
    }
private:
    const std::vector<std::vector<double>>& X;
    const std::vector<int>& y;
    int max_depth;
    int min_samples;
};

// Collector: gathers results into provided vector
class Collector final : public ff_node {
public:
    Collector(std::vector<DecisionTree*>& outvec, std::mutex& m) : out(outvec), mtx(m) {}
    void* svc(void* task) override {
        const auto* r = static_cast<Result*>(task);
        {
            std::lock_guard<std::mutex> lk(mtx);
            out[r->tree_id] = r->tree_ptr;
        }
        delete r;
        return GO_ON;
    }
private:
    std::vector<DecisionTree*>& out;
    std::mutex& mtx;
};

// train forest using FastFlow farm
std::vector<std::unique_ptr<DecisionTree>> train_forest_fastflow(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    int n_trees,
    int max_depth,
    int min_samples,
    size_t sample_size,
    unsigned int base_seed,
    int n_workers = 0) // 0 => auto
{
    if (sample_size == 0) sample_size = X.size();
    // prepare tasks
    std::queue<BuildTask> tasks;
    for (int t = 0; t < n_trees; ++t) {
        unsigned int seed = base_seed + static_cast<unsigned int>(t);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<size_t> dist(0, X.size() - 1);
        BuildTask bt;
        bt.tree_id = t;
        bt.seed = seed;
        bt.sample_idx.reserve(sample_size);
        for (size_t i = 0; i < sample_size; ++i) bt.sample_idx.push_back(dist(rng));
        tasks.push(std::move(bt));
    }

    std::mutex task_mtx;
    std::vector<DecisionTree*> raw_out(n_trees, nullptr);
    std::mutex out_mtx;

    // build worker vector
    std::vector<std::unique_ptr<ff_node>> workers;
    if (n_workers <= 0) n_workers = std::min<int>(n_trees, std::thread::hardware_concurrency());
    for (int i = 0; i < n_workers; ++i)
        workers.emplace_back(std::make_unique<Worker>(X, y, max_depth, min_samples));

    Emitter emitter(tasks, task_mtx);
    Collector collector(raw_out, out_mtx);

    std::vector<ff_node*> wptrs;
    for (auto &w : workers) wptrs.push_back(w.get());

    ff_farm farm(std::move(wptrs), &emitter, &collector);
    farm.remove_emitter();
    farm.run_and_wait_end();

    // move into unique_ptr vector
    std::vector<std::unique_ptr<DecisionTree>> forest;
    forest.reserve(n_trees);
    for (int i = 0; i < n_trees; ++i) forest.emplace_back(raw_out[i]);

    return forest;
}
