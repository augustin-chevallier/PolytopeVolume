#pragma once

#include "Polytope.hpp"
#include "eval_exp.hpp"
#include <random>
#include <iostream>




class GaussianSamplerBallWalk {
public:

	GaussianSamplerBallWalk() {
		std::random_device rd;
		rng = std::default_random_engine(rd());
	}

	Eigen::VectorXd randn(int size) {
		Eigen::VectorXd x(size);
		std::normal_distribution<> d{ 0,1 };
		for (int i = 0; i < size; i++) {
			x[i] = d(rng);
		}
		return x;
	}

	double unif01() {
		std::uniform_real_distribution<double> u(0, 1);
		return u(rng);
	}


	// a step of the ball walk
	Eigen::VectorXd ballWalk(Polytope K, Eigen::VectorXd x, double a, double delta) {
		Eigen::VectorXd next_pt = x;
		double f_x = gaussian_function(x, a);
		Eigen::VectorXd u = randn(K.dim);
		Eigen::VectorXd y = x + delta * u / u.norm() * std::pow(unif01(), (1. / K.dim));
		if (K.in_K(y)) {
			double f_y = gaussian_function(y, a);
			double pr = unif01();
			if (pr <= f_y / f_x) {
				next_pt = y;
			}
		}
		return next_pt;
	}


	// generate the next point from x \in K according to some pre - defined random walk
	Eigen::VectorXd getNextPoint(Polytope K, Eigen::VectorXd x, double a) {
		double delta = 4 * K.r / sqrt(std::max(1., a) * K.dim);
		x = ballWalk(K, x, a, delta);
		return x;
	}

	std::default_random_engine rng;
};