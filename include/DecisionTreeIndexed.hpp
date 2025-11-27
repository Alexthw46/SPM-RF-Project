#pragma once
#include <vector>
#include <random>
#include <unordered_map>
#include "Node.hpp"


/**
 * @brief Lightweight view wrapper for column-major data stored in a flat array.
 *
 * Provides read-only access to a 2D dataset stored in column-major layout
 * using a single contiguous array.
 */
struct ColMajorViewFlat {
    /** Pointer to the data in column-major order: outer index = feature, inner index = sample. */
    const double *data;

    /** Number of samples (rows). */
    size_t n_samples;

    /** Number of features (columns). */
    size_t n_features;

    /**
     * @brief Access the value for sample `i` and feature `f`.
     * @param i Sample index (row in logical row-major view).
     * @param f Feature index (column).
     * @return The value at data[f * n_samples + i].
     */
    double operator()(const size_t i, const size_t f) const {
        return data[f * n_samples + i];
    }
};

/**
 * @brief Lightweight view wrapper for column-major data.
 *
 * Provides read-only access to a 2D dataset stored in column-major layout.
 * Used to access a transposed row-major dataset without changing the original access patterns.
 * (outer vector indexes features, inner vector indexes samples).
 */
struct ColMajorView {
    /** Reference to column-major data: outer index = feature, inner index = sample. */
    const std::vector<std::vector<double> > &data;
    size_t n_samples;
    size_t n_features;
    /**
     * @brief Access the value for sample `i` and feature `f`.
     * @param i Sample index (row in a logical row-major view).
     * @param f Feature index (column).
     * @return The value at data[f][i].
     */
    double operator()(const size_t i, const size_t f) const { return data[f][i]; }
};

struct RowMajorView {
    const std::vector<std::vector<double> > &data;
    size_t n_samples;
    size_t n_features;
    double operator()(const size_t i, const size_t f) const { return data[i][f]; }
};

class DecisionTree {
public:
    DecisionTree(int max_depth_, int min_samples_, unsigned int seed);

    template<class View>
    void fit(const View &Xc,
             const std::vector<int> &y,
             const std::vector<size_t> &indices) {
        // Start the recursive build from root using column-major view
        root = build(Xc, y, indices, 0);

        // Find the bound to reserve space for flat representation
        const size_t bound1 = (static_cast<size_t>(1) << (max_depth + 1)) - 1;
        const size_t bound2 = 2 * indices.size() - 1;

        // Create flat representation for prediction
        const size_t reserve_size = std::min(bound1, bound2);
        flat.clear();
        flat.reserve(reserve_size);
        dfs_flat(root, flat);
        has_flat = true;
    }

    void set_flat(std::vector<FlatNode> &&flat_tree) {
        flat = std::move(flat_tree);
        has_flat = true;
    }

    static int predict_one(const Node *node, const std::vector<double> &x);

    [[nodiscard]] int predict_flat(const std::vector<double> &x) const;

    [[nodiscard]] int predict(const std::vector<double> &x) const {
        if (has_flat)
            return predict_flat(x);
        return predict_one(root, x);
    }

    Node *root = nullptr;

    /**
     * @brief Serialize a tree into a flat vector of \c FlatNode entries using a preorder traversal.
     *
     * This function appends a representation of the subtree rooted at \p node into \p out
     * and returns the index within \p out where the current node was placed.
     *
     * The function writes the following fields into each appended \c FlatNode (in order):
     *  - \c is_leaf   : whether the node is a leaf
     *  - \c feature   : splitting feature index (undefined for leaves)
     *  - \c threshold : splitting threshold (undefined for leaves)
     *  - \c label     : predicted label for leaf nodes (undefined for internal nodes)
     *  - \c right     : index of the right child in \p out (or -1 for leaves)
     *
     * Notes on layout:
     *  - The left child is serialized immediately after the parent (so its index is parent_index + 1).
     *  - The right child is serialized after the entire left subtree; its index is computed
     *    from the recursive call that serializes the right subtree.
     *
     * @param node Pointer to the root of the subtree to serialize (must not be null).
     * @param out  Vector to which \c FlatNode entries will be appended.
     * @return The index within \p out where the current node's \c FlatNode was stored.
     */
    static int dfs_flat(const Node *node, std::vector<FlatNode> &out) {
        const int me = out.size();
        out.emplace_back();

        auto &[is_leaf, feature, threshold, label, right] = out.back();
        is_leaf = node->is_leaf;
        feature = node->feature;
        threshold = node->threshold;
        label = node->label;

        if (node->is_leaf) {
            right = -1;
            return me;
        }

        // left child is exactly next, no need of tracking index
        dfs_flat(node->left, out);

        // right child: stored after left subtree
        const int right_idx = dfs_flat(node->right, out);

        right = right_idx;
        return me;
    }

    [[nodiscard]] std::vector<FlatNode> getFlat() const {
        return flat;
    }

    [[nodiscard]] bool hasFlat() const {
        return has_flat;
    }

private:
    int max_depth;
    int min_samples;
    std::mt19937 gen;

    // Flat representation
    std::vector<FlatNode> flat;
    bool has_flat = false;

    [[nodiscard]] static double gini_from_counts(const std::unordered_map<int, int> &counts, size_t n_features);

    [[nodiscard]] static double gini(const std::vector<int> &y);

    [[nodiscard]] static int majority_label_from_counts(const std::unordered_map<int, int> &counts);

    template<class View>
    Node *build(const View &Xc,
                const std::vector<int> &y,
                const std::vector<size_t> &indices,
                int depth);
};
