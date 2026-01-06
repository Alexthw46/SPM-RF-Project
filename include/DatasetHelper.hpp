#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <set>
#include <map>
#include <numeric>
#include <iomanip>

class DatasetHelper {
public:
    // Load CSV in X (features) e y (String labels mapped to int)
    static bool loadCSV(const std::string &filename, std::vector<std::vector<double> > &X, std::vector<int> &y) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        std::string line;
        // Skip header
        if (!std::getline(file, line)) return false;

        std::unordered_map<std::string, int> label_map;
        int label_counter = 0;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> features;
            std::vector<std::string> tokens;

            while (std::getline(ss, cell, ',')) tokens.push_back(cell);
            if (tokens.empty()) continue;

            std::string str_label = tokens.back();
            tokens.pop_back();

            if (!label_map.contains(str_label)) {
                label_map[str_label] = label_counter++;
            }
            int label = label_map[str_label];

            for (const auto &t: tokens) features.push_back(std::stod(t));

            X.push_back(features);
            y.push_back(label);
        }

        file.close();
        return true;
    }

    // Transpose the row-major data to column-major, as a copy
    static std::vector<std::vector<double> > transpose(const std::vector<std::vector<double> > &X) {
        if (X.empty()) return {};

        const size_t n_samples = X.size();
        const size_t n_features = X[0].size();
        std::vector X_col_major(n_features, std::vector<double>(n_samples));

        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                X_col_major[j][i] = X[i][j];
            }
        }

        return X_col_major;
    }

    // Transpose the row-major data to column-major, as a flat vector
    static std::vector<double> transpose_flat(const std::vector<std::vector<double> > &X) {
        if (X.empty()) return {};

        const size_t n_samples = X.size();
        const size_t n_features = X[0].size();
        std::vector<double> X_col_major(n_samples * n_features);

        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                X_col_major[j * n_samples + i] = X[i][j]; // column-major layout
            }
        }

        return X_col_major;
    }

    // Write integer vector to CSV (one value per line)
    static void writeToCSV(const char *str, const std::vector<int> &vector) {
        std::ofstream file(str);
        if (!file.is_open()) return;

        for (const auto &val: vector) {
            file << val << "\n";
        }

        file.close();
    };

    /**
     * @brief Split dataset into training and testing sets using indices.
     */
    static void split_indices(const size_t n_samples, const double test_size,
                              std::vector<size_t> &train_indices,
                              std::vector<size_t> &test_indices,
                              const bool shuffle = true,
                              const unsigned int seed = std::random_device{}()) {
        // Create indices [0, 1, 2, ..., n_samples-1]
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);

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

    static double f1_score(const std::vector<int> &y_true,
                           const std::vector<int> &y_pred) {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            return 0.0;
        }

        // collect labels
        std::set<int> labels_set;
        labels_set.insert(y_true.begin(), y_true.end());
        labels_set.insert(y_pred.begin(), y_pred.end());
        const std::vector<int> labels(labels_set.begin(), labels_set.end());

        std::map<int, int> tp, fp, fn, support;
        for (int lbl: labels) {
            tp[lbl] = fp[lbl] = fn[lbl] = support[lbl] = 0;
        }

        for (size_t i = 0; i < y_true.size(); ++i) {
            int t = y_true[i];

            if (int p = y_pred[i]; t == p) {
                ++tp[t];
            } else {
                ++fp[p];
                ++fn[t];
            }
            ++support[t];
        }

        double weighted_f1 = 0.0;
        int total_support = 0;

        for (int lbl: labels) {
            const int t = tp[lbl];
            const int fpv = fp[lbl];
            const int fnv = fn[lbl];
            const int sup = support[lbl];

            const double precision = (t + fpv > 0) ? static_cast<double>(t) / (t + fpv) : 0.0;
            const double recall = (t + fnv > 0) ? static_cast<double>(t) / (t + fnv) : 0.0;
            const double f1 = (precision + recall > 0.0)
                                  ? 2.0 * precision * recall / (precision + recall)
                                  : 0.0;

            weighted_f1 += f1 * sup;
            total_support += sup;
        }

        return (total_support > 0) ? weighted_f1 / total_support : 0.0;
    }

    /*
     * @brief Generate a classification report similar to sklearn's classification_report.
     */
    static std::string classification_report(const std::vector<int> &y_true,
                                             const std::vector<int> &y_pred,
                                             int width_name = 9,
                                             int width_val = 9,
                                             int float_precision = 2) {
        if (y_true.size() != y_pred.size() || y_true.empty()) {
            return {};
        }

        // collect labels
        std::set<int> labels_set;
        labels_set.insert(y_true.begin(), y_true.end());
        labels_set.insert(y_pred.begin(), y_pred.end());
        std::vector labels(labels_set.begin(), labels_set.end());

        // initialize counts
        std::map<int, int> tp, fp, fn, support;
        for (int lbl: labels) {
            tp[lbl] = fp[lbl] = fn[lbl] = support[lbl] = 0;
        }

        for (size_t i = 0; i < y_true.size(); ++i) {
            int t = y_true[i];
            if (int p = y_pred[i]; t == p) {
                ++tp[t];
            } else {
                ++fp[p];
                ++fn[t];
            }
            ++support[t];
        }

        // compute metrics per label
        struct Metrics {
            double precision;
            double recall;
            double f1;
            int sup;
        };
        std::map<int, Metrics> metrics;
        for (int lbl: labels) {
            int t = tp[lbl];
            int fpv = fp[lbl];
            int fnv = fn[lbl];
            int sup = support[lbl];

            double prec = (t + fpv > 0) ? static_cast<double>(t) / static_cast<double>(t + fpv) : 0.0;
            double rec = (t + fnv > 0) ? static_cast<double>(t) / static_cast<double>(t + fnv) : 0.0;
            double f1 = (prec + rec > 0.0) ? 2.0 * prec * rec / (prec + rec) : 0.0;

            metrics[lbl] = {prec, rec, f1, sup};
        }

        // averages
        double macro_p = 0.0, macro_r = 0.0, macro_f1 = 0.0;
        double weighted_p = 0.0, weighted_r = 0.0, weighted_f1 = 0.0;
        int total_support = std::accumulate(support.begin(), support.end(), 0,
                                            [](const int acc, const std::pair<const int, int> &kv) {
                                                return acc + kv.second;
                                            });

        for (int lbl: labels) {
            auto [precision, recall, f1, sup] = metrics[lbl];
            macro_p += precision;
            macro_r += recall;
            macro_f1 += f1;

            weighted_p += precision * sup;
            weighted_r += recall * sup;
            weighted_f1 += f1 * sup;
        }

        if (int n_labels = static_cast<int>(labels.size()); n_labels > 0) {
            macro_p /= n_labels;
            macro_r /= n_labels;
            macro_f1 /= n_labels;
        }
        if (total_support > 0) {
            weighted_p /= total_support;
            weighted_r /= total_support;
            weighted_f1 /= total_support;
        }

        // format output
        std::ostringstream out;
        out << std::fixed << std::setprecision(float_precision);

        out << std::setw(width_name) << " " << " "
                << std::setw(width_val) << "precision" << " "
                << std::setw(width_val) << "recall" << " "
                << std::setw(width_val) << "f1-score" << " "
                << std::setw(width_val) << "support" << "\n\n";

        for (int lbl: labels) {
            auto [precision, recall, f1, sup] = metrics[lbl];
            out << std::setw(width_name) << lbl << " "
                    << std::setw(width_val) << precision << " "
                    << std::setw(width_val) << recall << " "
                    << std::setw(width_val) << f1 << " "
                    << std::setw(width_val) << sup << "\n";
        }

        out << "\n" << std::setw(width_name) << "macro avg" << " "
                << std::setw(width_val) << macro_p << " "
                << std::setw(width_val) << macro_r << " "
                << std::setw(width_val) << macro_f1 << " "
                << std::setw(width_val) << total_support << "\n";

        out << std::setw(width_name) << "weighted avg" << " "
                << std::setw(width_val) << weighted_p << " "
                << std::setw(width_val) << weighted_r << " "
                << std::setw(width_val) << weighted_f1 << " "
                << std::setw(width_val) << total_support << "\n";

        return out.str();
    }

};