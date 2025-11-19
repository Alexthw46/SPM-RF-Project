#include <bits/stdc++.h>
#include <iostream>
#include <mpi.h>
#include "RandomForest.hpp"
#include "CSVLoader.hpp"
using namespace std;

constexpr bool debug = false;

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<double>> X;
    vector<int> y;

    // Replace with path to your CSV
    const string csv_file = "../test/Iris.csv";

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

    // Create and train the random forest (MPI-aware version)
    RandomForest rf(5, 5, 3);
    rf.fit(X, y); // MPI-aware fit

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

