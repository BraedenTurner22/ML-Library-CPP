#include "classification-metrics.h"
#include <iostream>
#include <vector>

struct ConfusionMatrix {
    int TP, TN, FP, FN;
};

ConfusionMatrix calculateConfusionMatrix(const std::vector<int>& true_labels, 
                                           const std::vector<int>& predicted_labels) {
    ConfusionMatrix cm = {0, 0, 0, 0};
    
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (predicted_labels[i] == 1 && true_labels[i] == 1) cm.TP++;
        else if (predicted_labels[i] == 0 && true_labels[i] == 0) cm.TN++;
        else if (predicted_labels[i] == 1 && true_labels[i] == 0) cm.FP++;
        else cm.FN++;
    }
    
    return cm;
}