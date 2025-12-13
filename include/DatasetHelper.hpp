#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>

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

    static void writeToCSV(const char *str, const std::vector<int> &vector) {
        std::ofstream file(str);
        if (!file.is_open()) return;

        for (const auto &val: vector) {
            file << val << "\n";
        }

        file.close();
    };
};