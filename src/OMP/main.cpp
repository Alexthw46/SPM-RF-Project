#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "RandomForest.hpp"
#include "CSVLoader.hpp"
using namespace std;

constexpr bool debug = false;

int main() {
    vector<vector<double> > X;
    vector<int> y;

    // Replace with path to your CSV
    const string csv_file = "../test/Iris.csv";

    if (!CSVLoader::loadCSV(csv_file, X, y)) {
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
