#pragma once

#include <vector>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


double mean(const std::vector<double>& v)
{
    double sum = 0;

    for (auto& each : v)
        sum += each;

    return sum / v.size();
}

double var(const std::vector<double>& v)
{
    double square_sum_of_difference = 0;
    double mean_var = mean(v);
    auto len = v.size();

    double tmp;
    for (auto& each : v) {
        tmp = each - mean_var;
        square_sum_of_difference += tmp * tmp;
    }

    return square_sum_of_difference / (len - 1);
}

double gaussian_function(const Eigen::VectorXd& x, double a_i) {
    return exp(-a_i * x.squaredNorm());
}