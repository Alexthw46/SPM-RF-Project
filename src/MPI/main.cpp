#include <bits/stdc++.h>
#include <iostream>
#include <mpi.h>
#include "CSVLoader.hpp"
#include "RandomForest.hpp"

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

    // Create and train the random forest (MPI-aware version)
    RandomForest rf(n_trees, max_depth, max_label + 1, 0);
    rf.fit(X, y);

    if (rank == 0) cout << "Training completed.\n";
    // Evaluate accuracy (rank 0 can gather predictions)
    std::vector<int> predictions;
    if (rank == 0)
        predictions = rf.predict_batch(X); // returns only on rank 0

    if (rank == 0) {
        int correct = 0;
        for (size_t i = 0; i < predictions.size() && i < y.size(); ++i)
            if (predictions[i] == y[i]) ++correct;

        cout << "Accuracy: " << static_cast<double>(correct) / static_cast<double>(X.size()) << endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
