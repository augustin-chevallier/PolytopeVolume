#pragma once

//#include <cuda_runtime.h>
//#include <cublas_v2.h>

#include <Eigen/Dense>
#include <random>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>

//#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>


using MP_FLOAT = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<1000,long long>>;// boost::multiprecision::mpfr_float;

using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class Polytope {
public:

    bool in_K(const Eigen::VectorXd& x) {
        auto y = A * x - b;
        return (y.array() <= 0).all();
    }

    Polytope(Eigen::MatrixXd A_, Eigen::VectorXd b_) :A(A_), b(b_), dim(A_.cols()) {
        Abis = A;
        compute_min_ball();
    }

    void compute_min_ball() {
        std::vector<double> dists(A.rows());
        for (int i = 0; i < dists.size(); i++) {
            dists[i] = b(i) / A.row(i).norm();
        }
        max_r = *std::min_element(dists.begin(), dists.end());
    }

    MatrixRowMajor Abis;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

    Eigen::VectorXd point_inside;

    int dim;
    double max_r; //radius of the largest ball included in the polytope (centered at origin)

};


MP_FLOAT factorial(int n) {

    if (n == 0) return 1;
    else {
        return n * factorial(n - 1);
    }
    //return (n == 0) || (n == 1) ? 1 : n * factorial(n - 1);
}

std::pair<Eigen::MatrixXd,Eigen::MatrixXd> polar(Eigen::MatrixXd& A) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullU|Eigen::ComputeFullV);
    //A = W Sigma V*
    auto W = svd.matrixU();
    auto V = svd.matrixV();
    auto Sigma = svd.singularValues();

    auto U = W * V.transpose();
    //auto P = V * Sigma * V.transpose();
    //std::cout << "W" << W << std::endl << "V" << V << std::endl;
    return { U,V };
}

Eigen::MatrixXd randomRotation(int n) {
    Eigen::MatrixXd A(n, n);
    std::random_device rd;
    std::default_random_engine rng(rd());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A(i, j) = std::normal_distribution()(rng);
        }
    }


    auto res = polar(A);
    return res.first;
}


std::pair<Polytope, MP_FLOAT>  createPolytopeFromFile(std::string filenameA, std::string filenameB, bool applyRandomRotation = false) {

    std::ifstream file;
    file.open(filenameA);
    std::string str;
    std::vector<std::vector<double>> A_;
    while (std::getline(file, str)) {
        A_.push_back({});
        //std::cout << str << std::endl;
        std::istringstream sin(str);
        double tmp;
        while (sin >> tmp) {
            A_[A_.size() - 1].push_back(tmp);
        }
    }
    file.close();
    double dim = A_[0].size();
    double nHyperplanes = A_.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nHyperplanes, dim);
    for (int i = 0; i < nHyperplanes; i++) {
        for (int j = 0; j < dim; j++) {
            A(i, j) = A_[i][j];
        }
    }
    std::cout << A;
    Eigen::VectorXd b;
    b.resize(nHyperplanes);
    file.open(filenameB);
    int i = 0;
    while (std::getline(file, str)) {
        //std::cout << str << std::endl;
        std::istringstream sin(str);
        double tmp;
        while (sin >> tmp) {
            b[i] = tmp;
            i++;
        }
    }
    std::cout << b;
    Polytope p(A, b);
    MP_FLOAT true_vol = -1;
    return { p,true_vol };
}

std::pair<Polytope, MP_FLOAT>  createPolytope(std::string shape, int dim, bool applyRandomRotation = false, std::string A_file = "", std::string b_file = "") {


    Eigen::MatrixXd R;
    if (applyRandomRotation) {
        R = randomRotation(dim);
        //std::cout << "R" << R << std::endl;
    }

    if (shape == "cube") {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * dim, dim);
        for (int i = 0; i < dim; i++) {
            A(i, i) = 1;
            A(i + dim, i) = -1;
        }
        if (applyRandomRotation) A = A * R;
        Eigen::VectorXd b = Eigen::VectorXd::Ones(2 * dim);
        MP_FLOAT volume = std::pow(2, dim);
        return { Polytope(A, b),volume };
    }
    if (shape == "standard_simplex") {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dim+1, dim);
        for (int i = 0; i < dim; i++) {
            A(i, i) = -1;
            A(dim, i) = 1;
        }
        if (applyRandomRotation) A = A * R;
        Eigen::VectorXd b = Eigen::VectorXd::Ones(dim+1)/(dim+1.);
        MP_FLOAT volume = 1. / factorial(dim);
        return { Polytope(A,b),volume };
    }

    if (shape == "isotropic_simplex") {
        Eigen::MatrixXd x = Eigen::MatrixXd::Zero(dim, dim + 1);
        for (int i = 0; i < dim; i++) {
            x(i, i) = sqrt(1 - std::pow( x.col(i).norm(), 2));
            for (int j = i + 1; j < dim + 1; j++) {
                x(i, j) = (-1. / (double)dim - x.col(i).dot(x.col(j))) / x(i, i);
            }
        }

        Eigen::MatrixXd vol_mat = Eigen::MatrixXd::Zero(dim, dim);
        for (int i = 1; i < dim + 1; i++) {
            for (int j = 0; j < dim; j++) {
                vol_mat(i - 1, j) = x(j, i) - x(j, 0);
            }
        }

        MP_FLOAT volume = boost::multiprecision::pow(MP_FLOAT(dim), dim) / factorial(dim) * std::abs(vol_mat.determinant());

        return { Polytope(x.transpose(),Eigen::VectorXd::Ones(dim + 1)),volume };
    }
    if (shape == "file") {
        return createPolytopeFromFile(A_file, b_file);
    }

    std::cerr << "shape " << shape <<" not supported" << std::endl;
}
