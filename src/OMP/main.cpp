#include <bits/stdc++.h>
#include <iostream>
#include  <omp.h>
#include "RandomForestIndexed.hpp"
#include "CSVLoader.hpp"
#include "TrainTestSplit.hpp"
using namespace std;

int main(const int argc, char *argv[]) {
    bool debug = false;
    string csv_file = "../test/Iris.csv";
    int n_trees = 100; // default value
    int max_depth = 10; // default value
    int global_seed = 42; // default value

    for (int i = 1; i < argc; ++i) {
        if (string a = argv[i]; a == "-d" || a == "--debug") {
            debug = true;
        } else if (a == "-t" || a == "--trees") {
            if (i + 1 < argc) { n_trees = stoi(argv[++i]); }
        } else if (a == "-m" || a == "--max-depth") {
            if (i + 1 < argc) { max_depth = stoi(argv[++i]); }
        } else if (a == "-s" || a == "--seed") {
            if (i + 1 < argc) global_seed = stoi(argv[++i]);
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

    // Train-test split (80% train, 20% test)
    vector<size_t> train_indices, test_indices;
    TrainTestSplit::split_indices(X.size(), 0.2, train_indices, test_indices, true, global_seed);

    cout << "Train samples: " << train_indices.size()
            << ", Test samples: " << test_indices.size() << "\n";

    // Create training subsets
    const auto X_train = TrainTestSplit::subset_X(X, train_indices);
    const auto y_train = TrainTestSplit::subset_y(y, train_indices);

    // Create test subsets
    const auto X_test = TrainTestSplit::subset_X(X, test_indices);
    const auto y_test = TrainTestSplit::subset_y(y, test_indices);

    cout << "OpenMP variant using: " << omp_get_max_threads() << " devices\n";

    // Create and train the random forest
    RandomForest rf(n_trees, max_depth, max_label + 1, global_seed);
    rf.fit(X_train, y_train);

    if (debug) {
        // print the depth of each tree
        int i = 0;
        for (const auto& tree : rf.getForest()) {
            cout << "Tree " << i << " number of nodes: " << tree.getInfo() << "\n";
            ++i;
        }
    }

    // Evaluate accuracy on training set
    const auto train_predictions = rf.predict_batch(X_train);
    const double train_accuracy = TrainTestSplit::accuracy(train_predictions, y_train);
    cout << "Training Accuracy: " << train_accuracy << endl;
    cout << "Classification Report (Train):\n";
    cout << TrainTestSplit::classification_report(y_train, train_predictions) << endl;

    // Evaluate accuracy on test set
    const auto test_predictions = rf.predict_batch(X_test);
    const double test_accuracy = TrainTestSplit::accuracy(test_predictions, y_test);
    cout << "Test Accuracy: " << test_accuracy << endl;
    cout << "Classification Report (Test):\n";
    cout << TrainTestSplit::classification_report(y_test, test_predictions) << endl;
}
