#ifndef TRAINTESTSPLIT_HPP
#define TRAINTESTSPLIT_HPP

#include <vector>
#include <random>
#include <algorithm>

/**
 * @brief Utility class for splitting data into training and testing sets using indices.
 *
 * This implementation avoids copying data by working with index vectors.
 */
class TrainTestSplit {
public:
    /**
     * @brief Split dataset into training and testing sets using indices.
     *
     * @param n_samples Total number of samples in the dataset.
     * @param test_size Fraction of data to use for testing (0.0 to 1.0).
     * @param train_indices Output vector of training indices.
     * @param test_indices Output vector of testing indices.
     * @param shuffle Whether to shuffle the indices before splitting.
     * @param seed Random seed for shuffling.
     */
    static void split_indices(const size_t n_samples, const double test_size,
                              std::vector<size_t> &train_indices,
                              std::vector<size_t> &test_indices,
                              const bool shuffle = true,
                              const unsigned int seed = std::random_device{}()) {
        // Create indices [0, 1, 2, ..., n_samples-1]
        std::vector<size_t> indices(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            indices[i] = i;
        }

        // Shuffle if requested
        if (shuffle) {
            std::mt19937 gen(seed);
            std::ranges::shuffle(indices, gen);
        }

        // Calculate split point
        const auto test_count = static_cast<size_t>(static_cast<double>(n_samples) * test_size);
        const auto train_count = static_cast<long>(n_samples - test_count);

        // Split indices
        train_indices.assign(indices.begin(), indices.begin() + train_count);
        test_indices.assign(indices.begin() + train_count, indices.end());
    }

    /**
     * @brief Create subset of features based on indices (creates a copy).
     *
     * @param X Original feature matrix.
     * @param indices Indices to select.
     * @return Subset of X containing only rows specified by indices.
     */
    static std::vector<std::vector<double> > subset_X(
        const std::vector<std::vector<double> > &X,
        const std::vector<size_t> &indices) {
        std::vector<std::vector<double> > subset;
        subset.reserve(indices.size());
        for (const size_t idx: indices) {
            subset.push_back(X[idx]);
        }
        return subset;
    }

    /**
     * @brief Create subset of labels based on indices (creates a copy).
     *
     * @param y Original label vector.
     * @param indices Indices to select.
     * @return Subset of y containing only elements specified by indices.
     */
    static std::vector<int> subset_y(
        const std::vector<int> &y,
        const std::vector<size_t> &indices) {
        std::vector<int> subset;
        subset.reserve(indices.size());
        for (const size_t idx: indices) {
            subset.push_back(y[idx]);
        }
        return subset;
    }

    /**
     * @brief Calculate accuracy between predictions and true labels.
     *
     * @param predictions Predicted labels.
     * @param y_true True labels.
     * @return Accuracy as a double between 0.0 and 1.0.
     */
    static double accuracy(const std::vector<int> &predictions,
                           const std::vector<int> &y_true) {
        if (predictions.size() != y_true.size() || predictions.empty()) {
            return 0.0;
        }

        int correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == y_true[i]) {
                ++correct;
            }
        }

        return static_cast<double>(correct) / static_cast<double>(predictions.size());
    }
};

#endif // TRAINTESTSPLIT_HPP
