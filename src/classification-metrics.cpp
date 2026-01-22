#include "classification-metrics.h"
#include <iostream>
#include <vector>
#include <cmath>

//Functions to calculate standard classification metrics precision, recall, r1, accuracy

double calculate_precision(int TP, int FP) {
    if (TP + FP == 0) return 0.0;
    return static_cast<double>(TP) / (TP + FP);
}

double calculate_recall(int TP, int FN) {
    if (TP + FN == 0) return 0.0;
    return static_cast<double>(TP) / (TP + FN);
}

double calculate_f1(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2 * ((precision * recall) / (precision + recall));
}

double calculate_accuracy(int TP, int TN, int FP, int FN) {
    if (TP + TN + FP + FN == 0) return 0.0;
    return static_cast<double>(TP + TN) / (TP + TN + FP + FN);
}