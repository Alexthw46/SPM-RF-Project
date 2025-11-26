#include <bits/stdc++.h>
#include <iostream>
#include "RandomForestIndexed.hpp"
#include "CSVLoader.hpp"
using namespace std;

int main(const int argc, char *argv[]) {
    bool debug = false;
    string csv_file = "../test/Iris.csv";
    int n_trees = 100;    // default preserved from original code
    int max_depth = 10;   // default preserved from original code

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-d" || a == "--debug") {
            debug = true;
        } else if (a == "-t" || a == "--trees") {
            if (i + 1 < argc) { n_trees = stoi(argv[++i]); }
        } else if (a == "-m" || a == "--max-depth") {
            if (i + 1 < argc) { max_depth = stoi(argv[++i]); }
        } else {
            csv_file = a;
        }
    }
    vector<vector<double> > X;
    vector<int> y;

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
        const size_t to_print = min<size_t>(5, X.size());
        for (size_t i = 0; i < to_print; ++i) {
            cout << "Row " << i << ": [";
            for (size_t j = 0; j < X[i].size(); ++j) {
                cout << X[i][j];
                if (j + 1 < X[i].size()) cout << ", ";
            }
            cout << "] -> " << y[i] << "\n";
        }
    }
    // Infer number of classes from labels
    const int max_label = *ranges::max_element(y);
    cout << "Inferred number of classes: " << (max_label + 1) << "\n";

    // Create and train the random forest
    RandomForest rf(n_trees, max_depth, max_label + 1);
    rf.fit(X, y);

    // Evaluate accuracy on full dataset
    int correct = 0;
    const auto predictions = rf.predict_batch(X);
    for (size_t i = 0; i < predictions.size() && i < y.size(); ++i) {
        if (predictions[i] == y[i]) ++correct;
    }

    cout << "Accuracy: " << static_cast<double>(correct) / static_cast<double>(X.size()) << endl;
}
