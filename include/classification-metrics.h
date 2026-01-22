#ifndef CLASSIFICATION_METRICS_H
#define CLASSIFICATION_METRICS_H

double calculate_precision(int TP, int FP);
double calculate_recall(int TP, int FN);
double calculate_f1(double precision, double recall);
double calculate_accuracy(int TP, int TN, int FP, int FN);

#endif