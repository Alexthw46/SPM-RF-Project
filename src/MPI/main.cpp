#include <bits/stdc++.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include "DatasetHelper.hpp"
#include "RandomForestIndexed.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    bool debug = false;
    string csv_file = "../test/Iris.csv";
    int n_trees = 100; // default preserved
    int max_depth = 10; // default preserved
    int global_seed = 42; // default preserved

    for (int i = 1; i < argc; ++i) {
        if (string a = argv[i]; a == "-d" || a == "--debug") {
            debug = true;
        } else if (a == "-t" || a == "--trees") {
            if (i + 1 < argc) n_trees = stoi(argv[++i]);
        } else if (a == "-m" || a == "-md" || a == "--max-depth") {
            if (i + 1 < argc) max_depth = stoi(argv[++i]);
        } else if (a == "-s" || a == "--seed") {
            if (i + 1 < argc) global_seed = stoi(argv[++i]);
        } else {
            csv_file = a;
        }
    }

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0 && debug) {
        std::cout << "MPI initialized with thread support level: " << provided << "\n";
    }

    // Compute threads per rank
    int threads_per_rank = omp_get_max_threads();

    if (rank == 0)
        std::cout << "MPI ranks: " << size
                << ", threads per rank: " << threads_per_rank << std::endl;

    vector<vector<double> > X;
    vector<int> y;

    // Replace with path to your CSV
    if (!DatasetHelper::loadCSV(csv_file, X, y)) {
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

    // Infer number of classes from labels
    const int max_label = *ranges::max_element(y);
    if (rank == 0) cout << "Inferred number of classes: " << (max_label + 1) << "\n";

    // Train-test split (80% train, 20% test)
    vector<size_t> train_indices, test_indices;
    DatasetHelper::split_indices(X.size(), 0.2, train_indices, test_indices, true, global_seed);

    if (rank == 0) {
        cout << "Train samples: " << train_indices.size()
                << ", Test samples: " << test_indices.size() << "\n";
    }

    // Create training subsets
    const auto X_train = DatasetHelper::subset_X(X, train_indices);
    const auto y_train = DatasetHelper::subset_y(y, train_indices);

    // Create test subsets
    const auto X_test = DatasetHelper::subset_X(X, test_indices);
    // Only rank 0 needs the test labels
    const auto y_test = rank == 0 ? DatasetHelper::subset_y(y, test_indices) : vector<int>();

    // Free original data memory
    X.clear();
    X.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();

    // Create and train the random forest for this MPI process
    RandomForest rf(n_trees, max_depth, max_label + 1, global_seed);

    // Train the trees assigned to this rank
    long train_time = rf.fit(X_train, y_train);

    if (debug)
        cout << "Training completed on rank " << rank << "in " << train_time << " us.\n";

    /*
    if (rank == 0) cout << "Starting Inference.\n";

    // Evaluate accuracy on training set (rank 0)
    cout << "[Rank " << rank << "] Evaluating training accuracy...\n";
    const std::vector<int> train_predictions = rf.predict_batch(X_train, distribute_data_or_trees);
    if (rank == 0) {
        const double train_accuracy = DatasetHelper::accuracy(train_predictions, y_train);
        cout << "Training Accuracy: " << train_accuracy << endl;
    }*/

    // Evaluate accuracy on test set (rank 0)
    if (debug)
        cout << "[Rank " << rank << "] Evaluating test accuracy...\n";

    const std::vector<int> test_predictions = rf.predict_batch(X_test);
    if (rank == 0) {
        const double test_accuracy = DatasetHelper::accuracy(test_predictions, y_test);
        cout << "Test Accuracy: " << test_accuracy << endl;
        cout << "Classification Report (Test):\n";
        cout << DatasetHelper::classification_report(y_test, test_predictions) << endl;

        // Write predictions to file
        const string n_ranks = std::to_string(size);
        const std::string test_filename = std::string("test_predictions_") + n_ranks + ".csv";
        DatasetHelper::writeToCSV(test_filename.c_str(), test_predictions);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
