#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// Generate random labels that represent true values (0 or 1) for binary classification
std::vector<int> generate_truth_tables(int size) {
    std::vector<int> labels;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i=0; i < size; i++) {
        //Assign label 1 with probability 0.4, else 0
        labels.push_back(dis(gen) > 0.6 ? 1 : 0);
    }

    return labels;
}

// Generate random labels that represent predicted values (0 or 1) for binary classification
std::vector<double> generate_predicted_values(const std::vector<int>& truth_labels) {
    std::vector<double> probs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 0.3);

    for (int label : truth_labels) {
        //Add Gaussian noise
        double prob = label + dis(gen);
        //Bring probability to [0, 1]
        prob = std::max(0.0, std::min(1.0, prob));
        probs.push_back(prob);
    }
    return probs;
}

// Compute True Positive Rate (TPR) and False Positive RATE (FPR) for a range of thresholds to plot the ROC curve
