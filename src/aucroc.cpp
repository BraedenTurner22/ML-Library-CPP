#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// Generate random labels that represent true values (0 or 1) for binary classification
std::vector<int> generate_truth_labels(int size) {
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

// Generate predicted probabilities for each sample, adding some noise to the true label
std::vector<double> generate_predicted_probs(const std::vector<int>& truth_labels) {
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
std::pair<std::vector<double>, std::vector<double>> roc_curve(const std::vector<int>& truth_labels, const std::vector<double>& predicted_probs) {

    std::vector<double> thresholds;
    //Define thresholds from 0.0-1.0 in steps of 0.1
    for (int i=0; i<=10; i++) {
        thresholds.push_back(i*0.1);
    }

    std::vector<double> tprs, fprs;

    // For each threshold, calculate TPR and FPR
    for (double threshold : thresholds) {
        int tp = 0, fp = 0, tn = 0, fn = 0;

        for (size_t i = 0; i < truth_labels.size(); ++i) {
            // Predict positive if probability >= threshold
            if (predicted_probs[i] >= threshold) {
                if (truth_labels[i] == 1) {
                    tp++; // True positive
                } else {
                    fp++; // False positive
                }
            } else {
                if (truth_labels[i] == 1) {
                    fn++; // False negative
                } else {
                    tn++; // True negative
                }
            }
        }

        // Calculate TPR = TP / (TP + FN)
        tprs.push_back(static_cast<double>(tp) / (tp + fn));
        // Calculate FPR = FP / (TN + FP)
        fprs.push_back(static_cast<double>(fp) / (tn + fp));
    }

    return make_pair(tprs, fprs);
}

int main() {
    // Generate a dataset of 500 samples
    std::vector<int> truth_labels = generate_truth_labels(500);
    // Generate predicted probabilities for each sample
    std::vector<double> predicted_probs = generate_predicted_probs(truth_labels);

    // Compute TPR and FPR for each threshold
    auto [tprs, fprs] = roc_curve(truth_labels, predicted_probs);

    // Output the ROC curve points
    std::cout << "ROC Curve Points:" << std::endl;
    for (size_t i = 0; i < tprs.size(); ++i) {
        std::cout << "Threshold: " << 0.1 * i << ", TPR: " << tprs[i]
             << ", FPR: " << fprs[i] << std::endl;
    }

    return 0;
}