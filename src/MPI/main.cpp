#include <bits/stdc++.h>
#include <iostream>
#include <mpi.h>
#include "CSVLoader.hpp"
#include "RandomForestIndexed.hpp"
#include "TrainTestSplit.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    bool debug = false;
    string csv_file = "../test/Iris.csv";
    int n_trees = 100;   // default preserved
    int max_depth = 10;  // default preserved

    for (int i = 1; i < argc; ++i) {
        if (string a = argv[i]; a == "-d" || a == "--debug") {
            debug = true;
        } else if (a == "-t" || a == "--trees") {
            if (i + 1 < argc) n_trees = stoi(argv[++i]);
        } else if (a == "-m" || a == "--max-depth") {
            if (i + 1 < argc) max_depth = stoi(argv[++i]);
        } else {
            csv_file = a;
        }
    }

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI does not provide required thread support.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<double>> X;
    vector<int> y;

    // Replace with path to your CSV
    if (!CSVLoader::loadCSV(csv_file, X, y)) {
        if (rank == 0) cerr << "Failed to open CSV file.\n";
        MPI_Finalize();
        return 1;
    }

    if (X.empty()) {
        if (rank == 0) cerr << "No data loaded.\n";
        MPI_Finalize();
        return 1;
    }

    if (X.size() != y.size()) {
        if (rank == 0) {
            cout << "Loaded file: " << csv_file << "\n";
            cout << "Rows: " << X.size() << " Labels: " << y.size() << "\n";
            cerr << "Mismatch between features and labels sizes\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Print first few rows for debugging (only rank 0)
    if (rank == 0 && debug) {
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
    if (rank == 0) cout << "Inferred number of classes: " << (max_label + 1) << "\n";

    // Train-test split (80% train, 20% test)
    vector<size_t> train_indices, test_indices;
    TrainTestSplit::split_indices(X.size(), 0.2, train_indices, test_indices);

    if (rank == 0) {
        cout << "Train samples: " << train_indices.size()
             << ", Test samples: " << test_indices.size() << "\n";
    }

    // Create training subsets
    const auto X_train = TrainTestSplit::subset_X(X, train_indices);
    const auto y_train = TrainTestSplit::subset_y(y, train_indices);

    // Create test subsets
    const auto X_test = TrainTestSplit::subset_X(X, test_indices);
    const auto y_test = TrainTestSplit::subset_y(y, test_indices);

    // Create and train the random forest for this MPI process
    RandomForestReplicated rf(n_trees, max_depth, max_label + 1);

    // Train the trees assigned to this rank
    rf.fit(X_train, y_train);

    cout << "Training completed on rank " << rank << ".\n";

    if (rank == 0) {
        cout << "Gathering trees from all ranks...\n";
    }

    // Broadcast trained trees to all ranks via MPI
    rf.gather_all_trees(MPI_COMM_WORLD);

    // Check if all trees are gathered (only rank 0)
    if (rank == 4) {
        if (rf.is_full_and_flat()) {
            cout << "All trees gathered successfully across MPI ranks.\n";
        } else {
            cout << "Error: Not all trees were gathered correctly.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    if (rank == 0) cout << "Training completed.\n";

    // Evaluate accuracy on training set (rank 0)
    const std::vector<int> train_predictions = rf.predict_batch(X_train);
    if (rank == 0) {
        const double train_accuracy = TrainTestSplit::accuracy(train_predictions, y_train);
        cout << "Training Accuracy: " << train_accuracy << endl;
    }

    // Evaluate accuracy on test set (rank 0)
    const std::vector<int> test_predictions = rf.predict_batch(X_test);
    if (rank == 0) {
        const double test_accuracy = TrainTestSplit::accuracy(test_predictions, y_test);
        cout << "Test Accuracy: " << test_accuracy << endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}