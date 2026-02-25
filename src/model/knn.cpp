#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <map>
#include <stdexcept>

// ── tiny stats helpers ────────────────────────────────────────────────────────
double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double stdev(const std::vector<double>& v) {
    double m = mean(v);
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - m) * (x - m);
    return std::sqrt(sq_sum / (v.size() - 1));  // sample stdev
}

// ── types ─────────────────────────────────────────────────────────────────────
using Sample = std::vector<double>;
using Dataset = std::vector<Sample>;
using Labels = std::vector<int>;

struct SplitData {
    Dataset X_train, X_test;
    Labels  y_train, y_test;
};

class KNN {

private:
    Dataset X_train;
    Labels  y_train;

public:

    // loading any dataset assumes all columns besides the last are paremeters, and the last column is the label column
    std::pair<Dataset, Labels> load_dataset(const std::string& path, bool contains_header) {
        Dataset X;
        Labels y;

        std::ifstream file(path);
        if (!file.is_open()) throw std::runtime_error("Cannot open dataset: " + path);

        std::string line;

        // If csv file contains a header, skip the first line
        if (contains_header) std::getline(file, line);

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::vector<double> row;
            std::stringstream ss(line);
            std::string token;

            while (std::getline(ss, token, ',')) {
                // Add each column value to row vector, convert string to float
                row.push_back(std::stod(token));
            }

            Labels::value_type label = static_cast<int>(row.back());
            row.pop_back();

            X.push_back(row);
            y.push_back(label);
        }
        return {X, y};
    }

    // needed for normalization
    Sample find_max_features(const Dataset& X) {
        size_t num_features = X[0].size();
        Sample maxes(num_features, -1e18);
        for (const auto& row : X)
            for (size_t i = 0; i < num_features; i++)
                maxes[i] = std::max(maxes[i], row[i]);
        return maxes;
    }

    // needed for normalization
    Sample find_min_features(const Dataset& X) {
        size_t num_features = X[0].size();
        Sample mins(num_features, 1e18);
        for (const auto& row : X)
            for (size_t i = 0; i < num_features; ++i)
                mins[i] = std::min(mins[i], row[i]);
        return mins;
    }

    // normalize based on column min/max
    Dataset normalize_features(const Dataset& X, const Sample& maxes, const Sample& mins) {
        Dataset normalized;
        for (const auto& row : X) {
            Sample normalized_row(row.size());
            for (size_t i = 0; i < row.size(); ++i)
                normalized_row[i] = (row[i] - mins[i]) / (maxes[i] - mins[i]);
            normalized.push_back(normalized_row);
        }
        return normalized;
    }

    // shuffle data set
    Dataset shuffle_dataset(Dataset& X, Labels& y) {
        std::vector<std::pair<Sample, int>> combined;
        // Pre-allocate memory for combined vector, as we already know the size
        combined.reserve(X.size());
        for (size_t i = 0; i < X.size(); ++i)
            combined.emplace_back(X[i], y[i]);
        
        std::shuffle(combined.begin(), combined.end(), std::mt19937{std::random_device{}()});

        for (size_t i = 0; i < combined.size(); ++i) {
            X[i] = combined[i].first;
            y[i] = combined[i].second;
        }
    }

    // train/test split
    SplitData split_data(const Dataset& X, const Labels& y, double test_size = 0.2) {
        size_t total = X.size();
        size_t test_count = static_cast<size_t>(total * test_size);

        SplitData sd;
        sd.X_test = Dataset(X.begin(), X.begin() + test_count);
        sd.y_test = Labels(y.begin(), y.begin() + test_count);
        sd.X_train = Dataset(X.begin() + test_count, X.end());
        sd.y_train = Labels(y.begin() + test_count, y.end());

        return sd;
    }

    double euclidean_distance(const Sample&a, const Sample&b) {
        double distance = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            distance += (a[i] - b[i]) * (a[i] - b[i]);
        return std::sqrt(distance);
    }

    void fit(const Dataset& X_train, const Labels& y_train) {
        this->X_train = X_train;
        this->y_train = y_train;
    }
};