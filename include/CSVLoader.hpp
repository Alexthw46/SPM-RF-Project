#ifndef CSVLOADER_HPP
#define CSVLOADER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>

class CSVLoader {
public:
    // Load CSV in X (features) e y (String labels mapped to int)
    static bool loadCSV(const std::string &filename, std::vector<std::vector<double>> &X, std::vector<int> &y) {
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

            for (const auto &t : tokens) features.push_back(std::stod(t));

            X.push_back(features);
            y.push_back(label);
        }

        file.close();
        return true;
    }
};

#endif // CSVLOADER_HPP

