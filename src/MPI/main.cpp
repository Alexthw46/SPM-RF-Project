#include <bits/stdc++.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include "DatasetHelper.hpp"
#include "RandomForestIndexed.hpp"
#include "TrainTestSplit.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    bool debug = false;
    string csv_file = "../test/Iris.csv";
    int n_trees = 100; // default preserved
    int max_depth = 10; // default preserved
    int global_seed = 42; // default preserved
    bool distribute_data_or_trees = true;

    for (int i = 1; i < argc; ++i) {
        if (string a = argv[i]; a == "-d" || a == "--debug") {
            debug = true;
        } else if (a == "-t" || a == "--trees") {
            if (i + 1 < argc) n_trees = stoi(argv[++i]);
        } else if (a == "-md" || a == "--max-depth") {
            if (i + 1 < argc) max_depth = stoi(argv[++i]);
        } else if (a == "-s" || a == "--seed") {
            if (i + 1 < argc) global_seed = stoi(argv[++i]);
        } else if (a == "-m" || a == "--mode") {
            // parse either d or t
            if (i + 1 < argc) {
                if (string mode = argv[++i]; mode == "d") {
                    distribute_data_or_trees = true;
                } else if (mode == "t") {
                    distribute_data_or_trees = false;
                }
            }
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

    if (rank == 0) {
        if (provided < MPI_THREAD_FUNNELED) {
            std::cerr << "MPI does not provide required thread support.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            std::cout << "MPI initialized with thread support level: " << provided << "\n";
        }
    }
    // Total logical cores on the node
    int total_cores = static_cast<int>(std::thread::hardware_concurrency());

    // Compute threads per rank, at least 1
    int threads_per_rank = total_cores; // std::max(1, total_cores / size);

    omp_set_num_threads(threads_per_rank);

    if (rank == 0)
        std::cout << "Total cores: " << total_cores
                << ", MPI ranks: " << size
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
    TrainTestSplit::split_indices(X.size(), 0.2, train_indices, test_indices, true, global_seed);

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
    RandomForestDistributed rf(n_trees, max_depth, max_label + 1, global_seed);

    // Train the trees assigned to this rank
    long train_time = rf.fit(X_train, y_train);

    if (debug)
        cout << "Training completed on rank " << rank << "in " << train_time << " us.\n";

    if (distribute_data_or_trees) {
        if (rank == 0) {
            cout << "Gathering trees from all ranks...\n";
        }

        if (size > 1)
            // Broadcast trained trees to all ranks via MPI
            rf.gather_all_trees(MPI_COMM_WORLD);

        // Check if all trees are gathered (only rank 0)
        if (rank == 0) {
            if (rf.is_full_and_flat()) {
                cout << "All trees gathered successfully in rank " << rank << ".\n";
                if (debug) {
                    auto forest = rf.getForest();
                    for (size_t i = 0; i < forest.size(); ++i) {
                        const auto &tree = forest[i];
                        cout << "Rank " << rank << " Tree " << i << " has " << tree.getFlat().size() << " nodes. Hash: "
                                << DecisionTree::hash_tree(tree.getFlat()) << "\n";
                    }
                }
            } else {
                cout << "Error: Not all trees were gathered correctly.\n";
                auto forest = rf.getForest();
                // find out which indexes are missing or are not flat
                for (size_t i = 0; i < forest.size(); ++i) {
                    if (const auto &tree = forest[i]; !tree.hasFlat()) {
                        cout << "Tree " << i << " is missing or not flat.\n";
                    }
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }
    /*
    if (rank == 0) cout << "Starting Inference.\n";

    // Evaluate accuracy on training set (rank 0)
    cout << "[Rank " << rank << "] Evaluating training accuracy...\n";
    const std::vector<int> train_predictions = rf.predict_batch(X_train, distribute_data_or_trees);
    if (rank == 0) {
        const double train_accuracy = TrainTestSplit::accuracy(train_predictions, y_train);
        cout << "Training Accuracy: " << train_accuracy << endl;
    }*/

    // Evaluate accuracy on test set (rank 0)
    if (debug)
        cout << "[Rank " << rank << "] Evaluating test accuracy...\n";

    const std::vector<int> test_predictions = rf.predict_batch(X_test, distribute_data_or_trees);
    if (rank == 0) {
        const double test_accuracy = TrainTestSplit::accuracy(test_predictions, y_test);
        cout << "Test Accuracy: " << test_accuracy << endl;
        cout << "Classification Report (Test):\n";
        cout << TrainTestSplit::classification_report(y_test, test_predictions) << endl;

        // Write predictions to file
        const string mode_str = distribute_data_or_trees ? "data_distributed" : "trees_distributed";
        const string n_ranks = std::to_string(size);
        const std::string test_filename = std::string("test_predictions_") + mode_str + "_" + n_ranks + ".csv";
        DatasetHelper::writeToCSV(test_filename.c_str(), test_predictions);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}