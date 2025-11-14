#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "RandomForest.hpp"
using namespace std;

constexpr bool debug = false;

// Helper function to load Iris CSV
bool loadCSV(const string &filename, vector<vector<double> > &X, vector<int> &y) {
    ifstream file(filename);
    if (!file.is_open()) return false;

    string line;
    // Skip header
    if (!getline(file, line)) return false;

    // Automap the string labels to integer classes
    unordered_map<string, int> label_map;
    int label_counter = 0;

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> features;

        vector<string> tokens;
        while (getline(ss, cell, ',')) tokens.push_back(cell);

        // Last column is the label (string)
        string str_label = tokens.back();
        tokens.pop_back();

        // Map string labels to integers
        if (!label_map.contains(str_label)) {
            label_map[str_label] = label_counter++;
        }
        int label = label_map[str_label];

        // Convert remaining tokens to features
        for (const auto &t: tokens) features.push_back(stod(t));

        X.push_back(features);
        y.push_back(label);
    }

    file.close();
    return true;
}

int main() {
    vector<vector<double> > X;
    vector<int> y;

    // Replace with path to your CSV
    const string csv_file = "../test/Iris.csv";

    if (!loadCSV(csv_file, X, y)) {
        cerr << "Failed to open CSV file.\n";
        return 1;
    }

    if (X.empty()) {
        cerr << "No data loaded.\n";
        return 1;
    }

    if (X.size() != y.size()) {
        cout << "Loaded file: " << csv_file << "\n";
        cout << "Rows: " << X.size() << " Labels: " << y.size() << "\n";
        cerr << "Mismatch between features and labels sizes\n";
    }

    // Print first few rows for debugging
    if (debug) {
        size_t to_print = min<size_t>(5, X.size());
        for (size_t i = 0; i < to_print; ++i) {
            cout << "Row " << i << ": [";
            for (size_t j = 0; j < X[i].size(); ++j) {
                cout << X[i][j];
                if (j + 1 < X[i].size()) cout << ", ";
            }
            cout << "] -> " << y[i] << "\n";
        }
    }

    // Create and train the random forest
    RandomForest rf(5, 5, 3);
    rf.fit(X, y);

    // Evaluate accuracy on full dataset
    int correct = 0;
    const auto predictions = rf.predict_batch(X);
    for (size_t i = 0; i < predictions.size() && i < y.size(); ++i) {
        if (predictions[i] == y[i]) ++correct;
    }

    cout << "Accuracy: " << static_cast<double>(correct) / static_cast<double>(X.size()) << endl;
}
