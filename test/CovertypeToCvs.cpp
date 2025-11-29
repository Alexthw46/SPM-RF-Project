//
// Created by alext on 28/11/2025.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <zlib.h>

auto INPUT_FILE = "../test/covtype.data.gz";
auto OUTPUT_FILE = "../test/covertype.csv";

int main() {
    gzFile gz = gzopen(INPUT_FILE, "rb");
    if (!gz) {
        std::cerr << "Cannot open input gz file\n";
        return 1;
    }

    std::ofstream out(OUTPUT_FILE);
    if (!out) {
        std::cerr << "Cannot open output CSV\n";
        return 1;
    }

    // Header generico
    out << "f1";
    for (int i = 2; i <= 54; ++i) out << ",f" << i;
    out << ",class\n";

    char buffer[8192];
    std::string line;

    while (gzgets(gz, buffer, sizeof(buffer)) != nullptr) {
        line = buffer;
        // Rimuove eventuali \n o \r
        if (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();

        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string token;
        bool first = true;
        for (int i = 0; i < 55; ++i) {  // 54 feature + 1 classe
            if (!(ss >> token)) break;
            if (!first) out << ",";
            out << token;
            first = false;
        }
        out << "\n";
    }

    gzclose(gz);
    out.close();

    std::cout << "Conversione completata: " << OUTPUT_FILE << "\n";
    return 0;
}
