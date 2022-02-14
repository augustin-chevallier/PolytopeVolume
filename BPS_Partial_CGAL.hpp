#pragma once

#include "Polytope.hpp"
#include <random>
#include <iostream>
#include <chrono>
#include "ESS.hpp"
#define CGAL_USE_MPFI
#include <CGAL/Cartesian_d.h>
#include <CGAL/Homogeneous_d.h>
#include <CGAL/intersections_d.h>
#include <CGAL/double.h>
#include <CGAL/Gmpfi.h>
#include <CGAL/Gmpfr.h>

#include <CGAL/MP_Float.h>
#include <CGAL/Quotient.h>
#include <CGAL/Lazy_exact_nt.h>

#include <boost/multiprecision/mpfr.hpp>

typedef CGAL::MP_Float     RT;
//typedef CGAL::Quotient<RT> FT;
//typedef CGAL::Lazy_exact_nt <CGAL::Quotient<RT>> FT;
//typedef CGAL::Gmpfi FT;
typedef boost::multiprecision::mpfr_float FT;

//typedef CGAL::Homogeneous_d<RT> Kernel;
typedef CGAL::Cartesian_d<FT> Kernel;
typedef CGAL::Point_d<Kernel>      Point;
typedef CGAL::Vector_d<Kernel>     Vector;
typedef CGAL::Direction_d<Kernel>  Direction;
typedef CGAL::Hyperplane_d<Kernel> Hyperplane;
typedef CGAL::Segment_d<Kernel>    Segment;
typedef CGAL::Ray_d<Kernel>        Ray;
typedef CGAL::Line_d<Kernel>       Line;

typedef Kernel::Squared_distance_d                                          Squared_distance_d;

using sec = std::chrono::duration<double>;


FT dot(std::vector<FT>& x_CGAL, std::vector<FT>& v_CGAL) {
	FT res = 0;
	for (int i = 0; i < x_CGAL.size(); i++) {
		res = res + x_CGAL[i] * v_CGAL[i];
	}
	return res;
}

std::vector<FT> matProd(Eigen::Matrix<FT, -1, -1>& A_CGAL,std::vector<FT>& x_CGAL) {
	std::vector<FT> Ax_CGAL(A_CGAL.rows());
	for (int i = 0; i < A_CGAL.rows(); i++) {
		Ax_CGAL[i] = 0;
		for (int j = 0; j < x_CGAL.size(); j++) {
			Ax_CGAL[i] = Ax_CGAL[i] + A_CGAL(i, j) * x_CGAL[j];
		}
	}
	return Ax_CGAL;
}

class GaussianSamplerBPSPartialCGAL {
public:

	struct Parameters {
		double output_rate = 1;
		double refresh_rate = 1;
	};

	GaussianSamplerBPSPartialCGAL() {
		std::random_device rd;
		rng = std::default_random_engine(rd());
		v_initialized = false;
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


	std::pair<FT, int> get_boundary_pt_CGAL(Polytope& K, std::vector<FT>& x_CGAL, std::vector<FT>& u_CGAL,int hyperplane_to_exclude) {

		for (int i = 0; i < x_CGAL.size(); i++) {
			//std::cout << "x " << x_CGAL[i] << std::endl; 
			//x_CGAL[i].normalize();
			//u_CGAL[i].normalize();
			//std::cout << "x " << x_CGAL[i] << std::endl;
		}

		Eigen::Matrix<FT, -1, -1> A_CGAL(K.A.rows(), K.A.cols());
		for (int i = 0; i < K.A.rows(); i++) {
			for (int j = 0; j < K.A.cols(); j++) {
				A_CGAL(i, j) = K.A(i, j);
			}
		}
		std::vector<FT> b_CGAL(K.b.size());
		for (int i = 0; i < K.b.size(); i++) {
			b_CGAL[i] = K.b(i);
		}
		/*std::vector<FT> u_CGAL(u.size());
		for (int i = 0; i < u.size(); i++) {
			u_CGAL[i] = u(i);
		}
		std::vector<FT> x_CGAL(u.size());
		for (int i = 0; i < x.size(); i++) {
			x_CGAL[i] = x(i);
		}*/

		std::vector<FT> Au_CGAL(K.b.size());
		std::vector<FT> Ax_CGAL(K.b.size());
		for (int i = 0; i < K.b.size(); i++) {
			Au_CGAL[i] = 0;
			Ax_CGAL[i] = 0;
			for (int j = 0; j < u_CGAL.size(); j++) {
				Au_CGAL[i] = Au_CGAL[i] + A_CGAL(i, j) * u_CGAL[j];
				Ax_CGAL[i] = Ax_CGAL[i] + A_CGAL(i, j) * x_CGAL[j];
			}
			//Ax_CGAL[i].normalize();
			//Au_CGAL[i].normalize();
			//std::cout << "Ax " << Ax_CGAL[i] << " Av " << Au_CGAL[i] << std::endl;
		}

		std::vector<FT> bAx_CGAL(K.b.size());
		for (int i = 0; i < K.b.size(); i++) {
			bAx_CGAL[i] = b_CGAL[i] - Ax_CGAL[i];
			//std::cout << "b " << bAx_CGAL[i] << " " << b_CGAL[i] << std::endl;
		}

		std::vector<FT> tmp_CGAL(K.b.size());
		for (int i = 0; i < K.b.size(); i++) {
			tmp_CGAL[i] = Au_CGAL[i] / bAx_CGAL[i];
		}
		FT m;
		int index_m;
		if (hyperplane_to_exclude != 0) {
			m = tmp_CGAL[0];
			index_m = 0;
		}
		else {
			m = tmp_CGAL[1];
			index_m = 1;
		}
		
		for (int i = index_m+1; i < K.b.size(); i++) {
			if (i != hyperplane_to_exclude) {
				//std::cout << "tmp " << tmp_CGAL[i] << " " << m << std::endl;
				//std::cout << "tmp " << CGAL::to_double(tmp_CGAL[i]) << " " << CGAL::to_double(m) << std::endl;
				if (m < tmp_CGAL[i]) {
					//std::cout << "min f " << m << " " <<tmp_CGAL[i] << std::endl;
					m = tmp_CGAL[i];
					index_m = i;
				}
				{
					//std::cout << "NO min f " << m << " " << tmp_CGAL[i] << std::endl;
				}
			}
		}
		FT t_min = 1 / tmp_CGAL[index_m];
		
		nOracles++;

		return { t_min,index_m };
	}

	std::pair<double,int> get_boundary_pt(Polytope& K, Eigen::VectorXd& x, Eigen::VectorXd& u, int hyperplane_to_exclude) {
	
		
		nOracles++;
		Eigen::VectorXd bAx = K.b - Ax;
		//Eigen::VectorXd Au = K.A * u;

		//std::cout << "AuAv " << (Ax - K.A*x).norm() << std::endl;

		auto tmp = Av.array() / bAx.array();

		int index_upper = -1;
		//double upper = 1 / tmp.maxCoeff(&index_upper);
		double upper;
		if (hyperplane_to_exclude != 0) {
			upper = tmp[0];
			index_upper = 0;
		}
		else {
			upper = tmp[1];
			index_upper = 1;
		}

		for (int i = index_upper + 1; i < K.b.size(); i++) {
			if (i != hyperplane_to_exclude) {
				//std::cout << "tmp " << tmp_CGAL[i] << " " << m << std::endl;
				//std::cout << "tmp " << CGAL::to_double(tmp_CGAL[i]) << " " << CGAL::to_double(m) << std::endl;
				if (upper < tmp[i]) {
					//std::cout << "min f " << m << " " <<tmp_CGAL[i] << std::endl;
					upper = tmp[i];
					index_upper = i;
				}
				{
					//std::cout << "NO min f " << m << " " << tmp_CGAL[i] << std::endl;
				}
			}
		}
		double t_min = 1 / tmp[index_upper];
		return { t_min,index_upper };
	}

	double getEvtTime(Eigen::VectorXd& x) {
		double a = v.dot(v);
		double b = x.dot(v);
		double u = unif01();

		double t = 0;

		if (b < 0)
			t = -b / a + std::sqrt(-2 * sigma * sigma / a * std::log(u));
		else
			t = -b / a + std::sqrt(b*b / (a*a) - 2 * sigma * sigma / a * std::log(u));
		return t;
	}

	FT getEvtTimeCGAL(std::vector<FT>& x_CGAL, std::vector<FT>& v_CGAL) {
		double a = (dot(v_CGAL,v_CGAL)).convert_to<double>();
		double b = (dot(x_CGAL,v_CGAL)).convert_to<double>();
		double u = unif01();

		FT t = 0;

		if (b < 0)
			t = -b / a + std::sqrt(-2 * sigma * sigma / a * std::log(u));
		else
			t = -b / a + std::sqrt(b * b / (a * a) - 2 * sigma * sigma / a * std::log(u));
		return t;
	}

	Eigen::VectorXd bounce(Eigen::VectorXd& v, Eigen::VectorXd& n, Eigen::VectorXd& Kn) {
		double c = 2 * v.dot(n) / n.dot(n);
		auto vr = v - c * n;
		Av = Av - c * Kn;
		return vr;
	}

	std::vector<FT> bounce_CGAL(std::vector<FT>& v_CGAL, std::vector<FT>& n_CGAL, std::vector<FT>& Kn_CGAL, std::vector<FT>& Av_CGAL) {
		FT c = 2 * dot(v_CGAL,n_CGAL) / dot(n_CGAL,n_CGAL);
		for (int i = 0; i < v_CGAL.size(); i++) {
			v_CGAL[i] = v_CGAL[i] - c * n_CGAL[i];
		}
		//auto vr = v - c * n;
		for (int i = 0; i < Av_CGAL.size(); i++) {
			Av_CGAL[i] = Av_CGAL[i] - c * Kn_CGAL[i];
		}
		//Av = Av - c * Kn;
		return v_CGAL;
	}

	double getResampleTime() {
		double u = unif01();
		return -std::log(u) / refresh_rate;
	}

	double getOutputTime() {
		double u = unif01();
		return -std::log(u) / output_rate;
	}

	double min(double a, double b, double c, double d) {
		return(std::min(std::min(a, b), std::min(c, d)));
	}

	FT min_CGAL(FT a, FT b, FT c, FT d) {
		FT m1;
		if (a < b) {
			m1 = a;
		}
		else {
			m1 = b;
		}
		FT m2;
		if (c < d) {
			m2 = c;
		}
		else {
			m2 = d;
		}
		if (m1 < m2) return m1;
		else return m2;
	}

	Parameters tune(Polytope& K, Eigen::VectorXd x, double a, bool tuning) {

		if (!An_initialized) {
			for (int i = 0; i < K.A.rows(); i++) {
				auto normal = K.A.row(i).transpose();
				An.push_back(K.A * normal);
			}
			An_initialized = true;
		}
		if (!v_initialized) {
			v = randn(x.size());
			Av = K.A * v;
			v_initialized = true;
		}
		sigma = std::sqrt(1. / 2. / a);

		nBounce = 0;
		nReflections = 0;
		nEvt = 0;

		refresh_rate = 1e-10; // essentially disable the refresh rate
		output_rate = 1e-10; //same for ouput rate

		int dim = x.size();

		double factor = 10;
		int nrepeat = 50;
		double t = 0;
		
		const auto before = std::chrono::steady_clock::now();

		Eigen::VectorXd oldx;
		for (int i = 0; i < nrepeat; i++) {
			v = randn(x.size());
			Av = K.A * v;
			oldx = x;
			double time = BPS(K, x, eps, dim * factor);
			while (!K.in_K(x)) {
				x = oldx;
				v = randn(x.size());
				Av = K.A * v;
				std::cout << "NOTINK";
				if (!K.in_K(x)) {
					std::cout << " maxi bug " << i;
				}
				time = BPS(K, x, eps, dim * factor);
			}
			t += time;
		}

		output_rate = 1/(t / factor / nrepeat);
		refresh_rate = 1/(t / factor/nrepeat);


		if (tuning) {
			const sec duration = std::chrono::steady_clock::now() - before;
			//std::cout << "tune first phase:" << duration.count() << std::endl;
			const auto before2 = std::chrono::steady_clock::now();

			std::cout << "tuning ESS" << std::endl;
			auto res = getESSPerSecInternal(K, x, a);

			double old_min = 0;
			double new_min = 0;
			double old_refresh_rate = 0;
			int niter = 0;
			do {
				niter++;
				old_refresh_rate = refresh_rate;
				old_min = std::min(res.first, res.second);
				if (res.first > res.second) {
					refresh_rate *= 2;
				}
				else {
					refresh_rate /= 2;
				}
				res = getESSPerSecInternal(K, x, a);
				new_min = std::min(res.first, res.second);
				std::cout << "new min " << new_min <<std::endl;
			} while (new_min > old_min && niter <= 10);
			refresh_rate = old_refresh_rate;

			const sec duration2 = std::chrono::steady_clock::now() - before2;
			//std::cout << "tune second phase:" << duration2.count() << std::endl;

			//std::cout << "done" << std::endl;

			//std::cout << "output rate " << output_rate << std::endl;
		}

		Parameters params;
		params.refresh_rate = refresh_rate;
		params.output_rate = output_rate;
		return params;
	}

	std::tuple<double, int, int, double> getESS(Polytope& K, Eigen::VectorXd& x, double a) {

		int dim = x.size();

		int nsamples = nSamplesESS;

		double res_ess = -1;
		double duration_s;

		while (res_ess < 0) {

			for (int k = 0; k < nsamples; k++) {
				x = getNextPoint(K, x, a);
			}

			Eigen::MatrixXd samples(nsamples, dim);
			Eigen::MatrixXd samples2(nsamples, 1);

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
			duration_s = duration.count();

			//std::cout << samples << std::endl;

			auto ess = ppl::math::ess<double>(samples);
			double min_ess = ess[0];
			for (int i = 0; i < dim; i++) {
				//std::cout << "ess i=" << i << " value = " << ess[i] / nsamples << std::endl;
				if (min_ess > ess[i]) min_ess = ess[i];
			}

			//std::cout << "ESS = " << min_ess/(double)nsamples << " for a = " << a << std::endl;

			auto ess2 = ppl::math::ess<double>(samples2);

			std::cout << "ESS result: " << min_ess << " " << ess2[0] << " nOracles " << nOracles << " " << duration_s << std::endl;

			res_ess = std::min(min_ess, ess2[0]);
			if (res_ess < 0) {
				std::cout << "Negative ESS, restarting with more points!" << std::endl;
				nSamplesESS = 2 * nSamplesESS;
			}
		}

		//std::cout << "ESS2 = " << ess2[0]/(double)nsamples << std::endl;

		return { res_ess,nsamples,nOracles, duration_s };
	}

	std::pair<double, double> getESSPerSecInternal(Polytope& K, Eigen::VectorXd& x, double a) {

		double min_ess = -1;
		double ess2 = -1;
		int dim = x.size();
		double duration_s;
		//Eigen::VectorXd x0 = Eigen::VectorXd::Zero(dim);

		while (min_ess < 0 || ess2 < 0) {

			int nsamples = nSamplesESS;

			for (int k = 0; k < nsamples; k++) {
				x = getNextPoint(K, x, a);
			}

			Eigen::MatrixXd samples(nsamples, dim);
			Eigen::MatrixXd samples2(nsamples, 1);

			const auto before = std::chrono::steady_clock::now();

			for (int k = 0; k < nsamples; k++) {
				x = getNextPoint(K, x, a);
				for (int i = 0; i < dim; i++) {
					samples(k, i) = x(i);
				}
				samples2(k, 0) = x.norm();
			}

			const sec duration = std::chrono::steady_clock::now() - before;
			duration_s = duration.count();
			//std::cout << samples << std::endl;

			auto ess = ppl::math::ess<double>(samples);
			min_ess = ess[0];
			for (int i = 0; i < dim; i++) {
				//std::cout << "ess i=" << i << " value = " << ess[i] / nsamples << std::endl;
				if (min_ess > ess[i]) min_ess = ess[i];
			}

			//std::cout << "ESS = " << min_ess / (double)nsamples << " for a = " << a << std::endl;

			ess2 = ppl::math::ess<double>(samples2)[0];

			if (std::min(min_ess, ess2) < 0) {
				std::cout << "Negative ESS, restarting with more points!" << std::endl;
				nSamplesESS = 2 * nSamplesESS;
			}
		}

		//std::cout << "ESS2 = " << ess2[0] / (double)nsamples << std::endl;

		return { min_ess / duration_s, ess2 / duration_s };
	}

	double getESSPerSec(Polytope& K, Eigen::VectorXd x, double a) {
		auto res = getESSPerSecInternal(K, x, a);
		return std::min(res.first, res.second);
	}

	double BPS_CGAL(Polytope& K, Eigen::VectorXd& x, double eps, int iter_max = -1) {

		//Ax = K.A * x;
		int last_hyperplane = -1;
		int d = x.size();
		Eigen::Matrix<FT, -1, -1> A_CGAL(K.A.rows(), K.A.cols());
		for (int i = 0; i < K.A.rows(); i++) {
			for (int j = 0; j < K.A.cols(); j++) {
				A_CGAL(i, j) = K.A(i, j);
			}
		}
		std::vector<FT> v_CGAL(d);
		for (int i = 0; i < d; i++) {
			v_CGAL[i] = v(i);
		}
		std::vector<FT> x_CGAL(d);
		//std::cout << " X BEGIN";
		for (int i = 0; i < x.size(); i++) {
			x_CGAL[i] = x(i);
			//std::cout << x_CGAL[i] << " ";// << CGAL::to_double(x_CGAL[i]) << " ";
		}

		std::vector<FT> Ax_CGAL(K.b.size());
		std::vector<FT> Av_CGAL(K.b.size());
		for (int i = 0; i < K.b.size(); i++) {
			Av_CGAL[i] = 0;
			Ax_CGAL[i] = 0;
			for (int j = 0; j < v_CGAL.size(); j++) {
				Av_CGAL[i] = Av_CGAL[i] + A_CGAL(i, j) * v_CGAL[j];
				Ax_CGAL[i] = Ax_CGAL[i] + A_CGAL(i, j) * x_CGAL[j];
			}
		}

		std::vector<std::vector<FT>> An_CGAL;
		for (int k = 0; k < K.A.rows(); k++) {
			std::vector<FT> normal(d);
			for (int i = 0; i < d; i++) {
				normal[i] = A_CGAL(k, i); // pull normal from
			}
			An_CGAL.push_back(matProd(A_CGAL ,normal));
		}

		FT t = 0;

		int nevt = 0;

		while (!(iter_max > 0 && nevt > iter_max)) {

			//t.normalize();
			//std::cout << "nevt" << nevt << " " << t << std::endl;
			nevt++;
			FT tau_0 = getEvtTimeCGAL(x_CGAL,v_CGAL);
			FT tau_1 = getResampleTime();
			std::pair ires = get_boundary_pt_CGAL(K, x_CGAL, v_CGAL,last_hyperplane);
			FT tau_2 = ires.first;
			FT tau_output = getOutputTime();
			//std::cout << "taus " << tau_0 << " " << tau_1 << " " << tau_2 << " " << tau_output << std::endl;
			int k = ires.second;
			//std::cout << "last H " << last_hyperplane << " k " << k << std::endl;

			//if (tau_2 < 0) {
				//std::cout << "A" << K.A << std::endl << "b" << K.b << std::endl;
				//std::cout << "bug, x = " << x.transpose() << std::endl << "v= " << v.transpose() << std::endl;
			//}

			/*if (tau_2 - eps < 0) {
				tau_2 = eps;
			}*/

			FT tau_evt = min_CGAL(tau_0, tau_1, tau_2, tau_output);  // we take tau_2 - eps to avoid going exactly on the boundary
			//FT tau_evt = tau_2;
			//std::cout << "tau_evt " << tau_evt << std::endl;
			for (int i = 0; i < d; i++) {
				x_CGAL[i] = x_CGAL[i] + tau_evt * v_CGAL[i];
				Ax_CGAL[i] = Ax_CGAL[i] + tau_evt * Av_CGAL[i];
			}
			//Ax = Ax + tau_evt * Av;
			t =  t + tau_evt;

			nEvt++;

			if (tau_evt == tau_output) {
				break;
			}


			//std::cout << "BEFORE " << (K.A * v - Av).norm() << std::endl;
			// work out if we are having a potential event, refreshment event, or boundary event
			if (tau_evt == tau_0) {  //potential event
				//Eigen::VectorXd Kn = K.A * x;
				last_hyperplane = -1;
				v_CGAL = bounce_CGAL(v_CGAL, x_CGAL, Ax_CGAL, Av_CGAL);
				nBounce++;
				//std::cout << "BOUNCE " << (K.A * v - Av).norm() << std::endl;
			}
			else { //bounce on the boundary
				if (tau_evt == tau_1) {  //refreshment event, n.b.velocity is Gaussian
					v = randn(d);
					for (int i = 0; i < d; i++) {
						v_CGAL[i] = v(i);
					}
					for (int i = 0; i < K.b.size(); i++) {
						Av_CGAL[i] = 0;
						for (int j = 0; j < v_CGAL.size(); j++) {
							Av_CGAL[i] = Av_CGAL[i] + A_CGAL(i, j) * v_CGAL[j];
						}
					}
					last_hyperplane = -1;
					//Av = K.A * v;
					//std::cout << "REFRESH " << (K.A * v - Av).norm() << std::endl;
				}
				else {
					nReflections++;
					std::vector<FT> normal(d);
					for (int i = 0; i < d; i++) {
						normal[i] = A_CGAL(k,i); // pull normal from
					}
					//std::cout << normal << std::endl;
					//std::cout << v << std::endl;
					FT c = 2 * dot(v_CGAL,normal) / dot(normal,normal);
					for (int i = 0; i < d; i++) {
						v_CGAL[i] = v_CGAL[i] - c * normal[i];
					}
					for (int i = 0; i < Av_CGAL.size(); i++) {
						Av_CGAL[i] = Av_CGAL[i] - c * An_CGAL[k][i];
					}
					last_hyperplane = k;
					// K.A* normal.transpose();
					//std::cout << "REFLECT " << (K.A*v - Av).norm() << std::endl;
				}
			}
		}

		for (int i = 0; i < d; i++) {
			x[i] = (x_CGAL[i]).convert_to<double>();
			v[i] = (v_CGAL[i]).convert_to<double>();
		}
		return t.convert_to<double>();
		//std::cout << "nevt " << nevt << std::endl;
	}

	double BPS(Polytope& K, Eigen::VectorXd& x, double eps, int iter_max = -1) {

		Ax = K.A * x;
		int last_hyperplane = -1;
		int d = x.size();

		double t = 0;
		
		int nevt = 0;

		while(!(iter_max > 0 && nevt > iter_max)) {

			nevt++;
			double tau_0 = getEvtTime(x);
			double tau_1 = getResampleTime();
			std::pair ires = get_boundary_pt(K, x, v, last_hyperplane);
			double tau_2 = ires.first;
			double tau_output = getOutputTime();
			int k = ires.second;

			//if (tau_2 < 0) {
				//std::cout << "A" << K.A << std::endl << "b" << K.b << std::endl;
				//std::cout << "bug, x = " << x.transpose() << std::endl << "v= " << v.transpose() << std::endl;
			//}

			if (tau_2 - eps < 0) {
				tau_2 = eps;
			}

			double tau_evt = min(tau_0, tau_1, tau_2 - eps,tau_output);  // we take tau_2 - eps to avoid going exactly on the boundary
			x = x + tau_evt * v;
			Ax = Ax + tau_evt * Av;
			t +=  tau_evt;
			
			nEvt++;

			if (tau_evt == tau_output) {
				break;
			}


			//std::cout << "BEFORE " << (K.A * v - Av).norm() << std::endl;
			// work out if we are having a potential event, refreshment event, or boundary event
			if (tau_evt == tau_0) {  //potential event
				//Eigen::VectorXd Kn = K.A * x;
				v = bounce(v, x,Ax);
				nBounce++;
				//std::cout << "BOUNCE " << (K.A * v - Av).norm() << std::endl;
			}
			else { //bounce on the boundary
				if (tau_evt == tau_1) {  //refreshment event, n.b.velocity is Gaussian
					v = randn(d);
					Av = K.A * v;
					//std::cout << "REFRESH " << (K.A * v - Av).norm() << std::endl;
				}
				else {
					nReflections++;
					auto normal = K.A.row(k); // pull normal from
					//std::cout << normal << std::endl;
					//std::cout << v << std::endl;
					double c = 2 * v.dot(normal) / normal.dot(normal);
					v = v - c * normal.transpose();
					Av = Av - c * An[k];// K.A* normal.transpose();
					//std::cout << "REFLECT " << (K.A*v - Av).norm() << std::endl;
				}
			}
		}

		return t;
		//std::cout << "nevt " << nevt << std::endl;
	}



	void setParameters(Parameters params) {
		refresh_rate = params.refresh_rate;
		output_rate = params.output_rate;
	}

	// generate the next point from x \in K according to some pre - defined random walk
	Eigen::VectorXd getNextPoint(Polytope& K, Eigen::VectorXd x, double a) {

		//std::cout << "x init " << x << std::endl;

		if (!An_initialized) {
			for (int i = 0; i < K.A.rows(); i++) {
				auto normal = K.A.row(i).transpose();
				An.push_back(K.A * normal);
			}
			An_initialized = true;
		}

		if (!v_initialized) {
			v = randn(x.size());
			Av = K.A * v;
			v_initialized = true;
		}
		sigma = std::sqrt(1. / 2. / a);

		Eigen::VectorXd oldx = x;
		Eigen::VectorXd oldv = v;
		Eigen::VectorXd oldAx = Ax;
		Eigen::VectorXd oldAv = Av;
		auto oldrng = rng;

		BPS(K, x, eps);
		/*std::cout <<"x ebd" << x << std::endl << std::endl;
		auto x_test = x;
		rng = oldrng;
		v = oldv;
		x = oldx;
		//Ax = oldAx;
		//Av = oldAv;
		BPS_CGAL(K, x, 0);
		std::cout << x << "old " << x_test << std::endl << std::endl;*/

		int max_precision = 2000;
		int current_precision = 128;
		while (!K.in_K(x) && current_precision < max_precision) {
			if (current_precision == 128) num_exit++;
			std::cout << "Escaped the convex, increasing precision to " << current_precision << std::endl;
			rng = oldrng;
			v = oldv;
			x = oldx;
			BPS_CGAL(K, x, 0);
		}


		int niter = 0;
		while (!K.in_K(x)) {
			if (niter == 0) num_exit_resample++;
			std::cout << "Precision increase was not suifisant, sampling a new velocity" << std::endl;
			if (!K.in_K(oldx)) {
				std::cout << "big bug " << std::endl;
			}
			niter++;
			std::cout << "not in K!!" << niter;
			//std::cout << "x " << x.transpose() << std::endl << "xold " << oldx.transpose() << std::endl;
			x = oldx;
			v = randn(x.size());
			Av = K.A * v;
			BPS(K, x, eps);
		}
		//std::cout << x.transpose() << std::endl;
		return x;
	}

	double eps = 0;// 1e-12;
	double sigma = 1;
	Eigen::VectorXd v;


	Eigen::VectorXd Ax;
	Eigen::VectorXd Av;

	std::vector<Eigen::VectorXd> An;
	bool An_initialized = false;

	double refresh_rate = 0.1;
	double output_rate = 0.5;

	int nSamplesESS = 100;

	std::default_random_engine rng;

	int nBounce = 0;
	int nReflections = 0;
	int nEvt = 0;
	uint64_t nOracles = 0;

	uint64_t num_exit = 0;
	uint64_t num_exit_resample = 0;

	bool ESS_estimation = true;

	bool v_initialized = false;
};