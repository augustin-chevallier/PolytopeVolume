#pragma once

#include "Polytope.hpp"
#include "Utilities.hpp"
#include "ESS.hpp"
#include <random>
#include <iostream>
#include <chrono>

using sec = std::chrono::duration<double>;


class GaussianSamplerHAR {
public:

	struct Parameters {
	};

	GaussianSamplerHAR() {
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

	double sign(double x) {
		if (x >= 0) {
			return 1.;
		}
		else {
			return -1;
		}
	}

	double get_max(Eigen::VectorXd& l, Eigen::VectorXd& u, double a_i) {
		// get the maximum value along the chord, which is the height of the bounding
		// box for rejection sampling
		auto a = -l;
		auto b = (u - l) / (u - l).norm();

		auto z = a.dot(b) * b + l;
		double low_bd = (l(1) - z(1)) / b(1);
		double up_bd = (u(1) - z(1)) / b(1);
		double ret1 = 0;
		if (sign(low_bd) == sign(up_bd)) {
			ret1 = std::max(gaussian_function(u, a_i), gaussian_function(l, a_i));
		}
		else {
			ret1 = gaussian_function(z, a_i);
		}
		return ret1;
	}

	// where the points are weighted by the distribution exp(-a_i || x || ^ 2)
	Eigen::VectorXd rand_exp_range(Eigen::VectorXd& l, Eigen::VectorXd& u, double a_i) {


		if (a_i > 1e-8 && (u - l).norm() >= 2 / std::sqrt(2 * a_i)) {
			// select from the 1d Gaussian chord if enough weight will be inside
			// K

			auto a = -l;
			auto b = (u - l) / (u - l).norm();
			auto z = a.dot(b) * b + l;
			double low_bd = (l(1) - z(1)) / b(1);
			double up_bd = (u(1) - z(1)) / b(1);

			double r = 0;
			while (true) {
				// sample from Gaussian along chord, and accept if inside(u, v)

				r = randn(1)[0] / std::sqrt(2 * a_i);
				if (r >= low_bd && r <= up_bd) {
					break;
				}
			}
			return r * b + z;
		}
		else {
			// otherwise do simple rejection sampling by a bounding rectangle
			double M = get_max(l, u, a_i);
			bool done = false;
			int its = 0;
			while (!done) {
				its = its + 1;
				double r = unif01();
				Eigen::VectorXd p = (1 - r) * l + r * u;
				double r_val = M * unif01();
				double fn = gaussian_function(p, a_i);
				if (r_val < fn) {
					done = true;
					return p;
				}
			}
		}
	}

	std::pair<Eigen::VectorXd, Eigen::VectorXd> get_boundary_pts(Polytope& K, Eigen::VectorXd& x, Eigen::VectorXd& u) {
		nOracles++;
		Eigen::VectorXd bAx = K.b - K.A * x;
		Eigen::VectorXd Au = K.A * u;

		auto temp = bAx.array() / Au.array();
		auto tmp = 1. / temp;

		double lower = 1 / tmp.minCoeff();
		double upper = 1 / tmp.maxCoeff();


		Eigen::VectorXd upperV = x + (upper - 1e-6) * u;
		Eigen::VectorXd lowerV = x + (lower + 1e-6) * u;
		return { upperV,lowerV };
	}

	Eigen::VectorXd hitAndRun(Polytope K, Eigen::VectorXd x, double a) {
		Eigen::VectorXd u = randn(K.dim);
		u = u / u.norm();
		auto [upper, lower] = get_boundary_pts(K, x, u);

		return rand_exp_range(lower, upper, a);
	}


	Eigen::VectorXd getNextPoint(Polytope K, Eigen::VectorXd x, double a) {
		x = hitAndRun(K, x, a);
		return x;
	}

	void setParameters(Parameters p) {};

	Parameters tune(Polytope& K, Eigen::VectorXd x, double a, bool t) {
		Parameters p;
		return p;
	}

	std::tuple<double, int, int, double> getESS(Polytope& K, Eigen::VectorXd x, double a) {

		int dim = x.size();
		//Eigen::VectorXd x0 = Eigen::VectorXd::Zero(dim);

		int nsamples = 1000;
		Eigen::MatrixXd samples(nsamples, dim);

		Eigen::MatrixXd samples2(nsamples, 1);

		for (int k = 0; k < nsamples; k++) {
			x = getNextPoint(K, x, a);
		}

		nOracles = 0;
		const auto before = std::chrono::steady_clock::now();
		for (int k = 0; k < nsamples; k++) {
			x = getNextPoint(K, x, a);
			for (int i = 0; i < dim; i++) {
				samples(k, i) = x(i);
			}
			samples2(k, 0) = x.norm();
		}
		const sec duration = std::chrono::steady_clock::now() - before;
		double duration_s = duration.count();
		//std::cout << samples << std::endl;

		auto ess = ppl::math::ess<double>(samples);
		double min_ess = ess[0];
		for (int i = 0; i < dim; i++) {
			//std::cout << "ess i=" << i << " value = " << ess[i] / nsamples << std::endl;
			if (min_ess > ess[i]) min_ess = ess[i];
		}

		std::cout << "ESS = " << min_ess / (double)nsamples << " for a = " << a << std::endl;

		auto ess2 = ppl::math::ess<double>(samples2);

		std::cout << "ESS2 = " << ess2[0] / (double)nsamples << std::endl;

		return { std::min(min_ess,ess2[0]),nsamples,nOracles, duration_s};
	}


	std::pair<double, double> getESSPerSecInternal(Polytope& K, Eigen::VectorXd x, double a) {

		int dim = x.size();
		Eigen::VectorXd x0 = Eigen::VectorXd::Zero(dim);

		int nsamples = 1000;// *dim;
		for (int k = 0; k < nsamples; k++) {
			x0 = getNextPoint(K, x0, a);
		}
		Eigen::MatrixXd samples(nsamples, dim);
		Eigen::MatrixXd samples2(nsamples, 1);

		const auto before = std::chrono::steady_clock::now();

		for (int k = 0; k < nsamples; k++) {
			x0 = getNextPoint(K, x0, a);
			for (int i = 0; i < dim; i++) {
				samples(k, i) = x0(i);
			}
			samples2(k, 0) = x0.norm();
		}

		const sec duration = std::chrono::steady_clock::now() - before;

		//std::cout << samples << std::endl;

		auto ess = ppl::math::ess<double>(samples);
		double min_ess = ess[0];
		for (int i = 0; i < dim; i++) {
			//std::cout << "ess i=" << i << " value = " << ess[i] / nsamples << std::endl;
			if (min_ess > ess[i]) min_ess = ess[i];
		}

		//std::cout << "ESS = " << min_ess / (double)nsamples << " for a = " << a << std::endl;

		auto ess2 = ppl::math::ess<double>(samples2);

		//std::cout << "ESS2 = " << ess2[0] / (double)nsamples << std::endl;

		return { min_ess / duration.count(), ess2[0] / duration.count() };
	}

	double getESSPerSec(Polytope& K, Eigen::VectorXd x, double a) {
		auto res = getESSPerSecInternal(K, x, a);
		return std::min(res.first, res.second);
	}


	std::default_random_engine rng;
	int nOracles = 0;
};