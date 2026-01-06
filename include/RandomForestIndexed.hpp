#pragma once
#include <algorithm>
#include <vector>
#include <random>

#include "DecisionTreeIndexed.hpp"

/**
 * @brief RandomForest classifier that aggregates multiple DecisionTree models.
 *
 * This class trains an ensemble of indexed decision trees and predicts class
 * labels by majority voting across the trees.
 */
class RandomForest {
public:
    /**
     * @brief Construct a new RandomForest object.
     * @param n_t Number of trees in the forest (default: 5).
     * @param max_depth Maximum depth for each decision tree (default: 5).
     * @param n_classes Number of target classes (default: 2).
     * @param seed Seed for the random number generator (default: randomized).
     */
    explicit RandomForest(int n_t = 5, int max_depth = 5, int n_classes = 2,
                          unsigned int seed = std::random_device{}());

    /**
     * @brief Train the random forest on the provided dataset.
     * @param X Feature matrix where each inner vector is a sample.
     * @param y Corresponding class labels for each sample in X.
     */
    long fit(const std::vector<std::vector<double> > &X, const std::vector<int> &y);

    /**
     * @brief Predict the class label for a single sample.
     * @param x Feature vector of the sample to predict.
     * @return Predicted class label.
     */
    [[nodiscard]] int predict(const std::vector<double> &x) const {
        if (trees.empty()) return -1;
        std::vector vote_count(n_classes, 0);
        for (const auto &t: trees) {
            if (const int p = t.predict(x); p >= 0 && p < n_classes) ++vote_count[p];
        }
        const auto it = std::ranges::max_element(vote_count);
        return static_cast<int>(std::distance(vote_count.begin(), it));
    }

    /**
     * @brief Predict class labels for a batch of samples.
     * @param X Feature matrix where each inner vector is a sample.
     * @return Vector of predicted class labels for each sample.
     */
    [[nodiscard]] std::vector<int> predict_batch(const std::vector<std::vector<double> > &X) const;

    [[nodiscard]] std::vector<DecisionTree> getForest() const {
        return trees;
    }

protected:
    /** @brief Number of trees in the ensemble. */
    int n_trees;
    /** @brief Maximum allowed depth for each decision tree. */
    int max_depth;
    /** @brief Number of distinct target classes. */
    int n_classes;
    /** @brief Collection of decision trees forming the forest. */
    std::vector<DecisionTree> trees;
    // @brief Random number generator seed. */
    unsigned int seed;
};

class VersatileRandomForest : public RandomForest {
public:
    using RandomForest::RandomForest;

    /**
 * @brief Train the random forest on the provided dataset.
 * @param X Feature matrix where each inner vector is a sample.
 * @param y Corresponding class labels for each sample in X.
 * @param parallelMode allows to select parallelization mode, 0 = none, 1 = OpenMP, 2 = FastFlow
 *
 * @return Time taken to fit the random forest model.
 */
    long fit(const std::vector<std::vector<double> > &X, const std::vector<int> &y, unsigned int parallelMode);

    /**
     * @brief Predict class labels for a batch of samples.
     * @param X Feature matrix where each inner vector is a sample.
     * @param parallelMode allows to select parallelization mode, 0 = none, 1 = OpenMP, 2 = FastFlow
     *
     * @return Vector of predicted class labels for each sample.
     */
    [[nodiscard]] std::vector<int> predict_batch(const std::vector<std::vector<double> > &X,
                                                 unsigned int parallelMode) const;
};
