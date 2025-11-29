#pragma once
#include <algorithm>
#include <mpi.h>
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
     * @param seed Seed for the random number generator (default: random device).
     */
    explicit RandomForest(int n_t = 5, int max_depth = 5, int n_classes = 2,
                          unsigned int seed = std::random_device{}());

    /**
     * @brief Train the random forest on the provided dataset.
     * @param X Feature matrix where each inner vector is a sample.
     * @param y Corresponding class labels for each sample in X.
     */
    void fit(const std::vector<std::vector<double> > &X, const std::vector<int> &y);

    /**
     * @brief Predict the class label for a single sample.
     * @param x Feature vector of the sample to predict.
     * @return Predicted class label.
     */
    [[nodiscard]] int predict(const std::vector<double> &x) const;

    /**
     * @brief Predict class labels for a batch of samples.
     * @param X Feature matrix where each inner vector is a sample.
     * @return Vector of predicted class labels for each sample.
     */
    [[nodiscard]] std::vector<int> predict_batch(const std::vector<std::vector<double> > &X) const;

    [[nodiscard]] bool is_full_and_flat() const {
        return std::ranges::all_of(trees, [](const DecisionTree &tree) {
            return tree.hasFlat() && !tree.getFlat().empty();
        });
    }

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
    /** @brief Mersenne Twister random number generator used for sampling. */
    std::mt19937 gen;
};

class RandomForestDistributed : public RandomForest {
public:
    using RandomForest::RandomForest;

    [[nodiscard]] std::vector<int> predict_batch(const std::vector<std::vector<double> > &X, bool distributionStrat) const;


    void gather_all_trees(MPI_Comm comm);
};

