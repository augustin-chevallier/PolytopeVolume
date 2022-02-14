#pragma once

#include "Polytope.hpp"
#include <random>
#include <iostream>
#include <chrono>

#include "ESS.hpp"


using sec = std::chrono::duration<double>;



class GaussianSamplerCS {
public:

	struct Parameters {
		double output_rate = 1;
		double refresh_rate = 1;
	};

	GaussianSamplerCS() {
		std::random_device rd;
		rng = std::default_random_engine(rd());
		v_initialized = false;
	}

	void randVelocity(int d) {
		v_index = std::uniform_int_distribution(0, d - 1)(rng);// np.random.randint(d);
		v_sign = std::uniform_int_distribution(0, 1)(rng) * 2 -1;
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


	std::pair<double, int> get_boundary_pt(Polytope& K,const Eigen::VectorXd& Ax) {
		auto Av =  v_sign* K.A.col(v_index);
		Eigen::VectorXd bAx = K.b - Ax;
		//Eigen::VectorXd Au = K.A * u;

		auto temp = bAx.array() / Av.array();
		auto tmp = 1. / temp;

		int index_upper = -1;
		double upper = 1 / tmp.maxCoeff(&index_upper);

		return { upper,index_upper };
	}

	double getEvtTime(Eigen::VectorXd& x) {
		double a = 1;// v.dot(v);
		double b = v_sign * x[v_index];// .dot(v);
		double u = unif01();

		double t = 0;

		if (b < 0) {
			t = -b / a + std::sqrt(-2 * sigma*sigma / a * std::log(u));
		}
		else {
			t = -b / a + std::sqrt(b * b / (a * a) - 2 * sigma*sigma / a * std::log(u));
		}
		return t;
	}

	std::vector<double> probaVelocity(Eigen::VectorXd& x) {

		double s = 0;
		int d = x.size();
		for (int i = 0; i < d; i++) {
			//ed = np.zeros(d);
			//ed[i] = 1;
			//s += abs(np.dot(x, ed));
			s += std::abs(x[i]);
		}
		s = s / sigma;
		std::vector<double> l(2 * d, 0);
		for (int i = 0; i < d; i++) {
			double ps = x[i];
			if (ps > 0) {
				l[2 * i] = 0;
				l[2 * i + 1] = ps / sigma / s;
			}
			else {
				l[2 * i] = -ps / sigma / s;
				l[2 * i + 1] = 0;
			}
		}
		return l;
	}

	void getNewVelocity(Eigen::VectorXd& x) {
		int d = x.size();
		std::vector<double> lv = probaVelocity(x);
		//ln = np.arange(2 * d);
		//std::discrete_distribution<int> distribution(lv.begin(),lv.end());
		//int i = distribution(rng);// np.random.choice(ln, p = lv);
		//Note: std::discrete_distribution is verryyyyy slow and does a bunch of copies and stuff, so here is my own version...
		double u = unif01();
		int i = 0;
		double acc = 0;
		for (i = 0; i < lv.size(); i++) {
			acc += lv[i];
			if (u < acc) break;
		}
		if (i % 2 == 0){
			v_index = i / 2;
			v_sign = 1;
		}
		else {
			v_index = i / 2;
			v_sign = -1;
		}
	}

	std::tuple< int,double,double > proposal(auto& normal) {
		int d = normal.size();
		int proposed_index = std::uniform_int_distribution(0, d - 1)(rng);// np.random.randint(d);
		double proposed_sign = -((normal[proposed_index] > 0)*2 -1);
		double correction = 1;
		return { proposed_index,proposed_sign, correction };
	}

	/*std::vector<double> getProbaVecProposal(int index,double sign, auto& normal) {
		int dim = normal.size();

		Eigen::VectorXd v = Eigen::VectorXd::Zero(dim);
		v[index] = sign;
		Eigen::VectorXd v_reflected = v - 2 * sign*normal[index] * normal.transpose() / normal.dot(normal);

		std::vector<double> proba_v(dim * 2,0);

		//std::cout << "normal" << normal.dot(normal) << std::endl;
		//double a = 0.1;

		double vrvr = v_reflected.dot(v_reflected);
		//std::cout << "refl " << v_reflected.transpose() << std::endl;
		//std::cout << vrvr << " ";

		double sum = 0;

		for (int i = 0; i < dim; i++) {
			if (normal[i] < 0) {
				double dist = 1 + vrvr - 2 * v_reflected[i];
				//std::cout << dist << " ";
				proba_v[i] = std::exp(-reflection_parameter * dist);
				sum += proba_v[i];
			}

		}
		for (int i = 0; i < dim; i++) {
			if (normal[i] > 0) {
				double dist = 1 + vrvr + 2 * v_reflected[i];
				//std::cout << dist << " ";
				proba_v[i + dim] = std::exp(-reflection_parameter * dist);
				sum += proba_v[i+dim];
			}

		}
		//std::cout << " \n";
		if (std::isnan(sum) || std::isinf(sum)) {
			std::cout << "BUG";
		}

		//double sum = std::accumulate(proba_v.begin(), proba_v.end(), 0);
		for (int i = 0; i < proba_v.size(); i++) {
			proba_v[i] /= sum;
			//std::cout << proba_v[i] << " ";
		}
		//std::cout << std::endl;
		
		return proba_v;
	}

	std::tuple< int, double, double > proposal_(auto& normal) {
		int d = normal.size();
		std::vector<double> proba_v = getProbaVecProposal(v_index,v_sign, normal);
		//i = np.random.choice(np.arange(2 * dim), p = proba_v);
		double u = unif01();
		int i = 0;
		double acc = 0;
		for (i = 0; i < proba_v.size(); i++) {
			acc += proba_v[i];
			if (u < acc) break;
		}
		int proposed_index = i;
		double proposed_sign = 1;
		if (i >= d) {
			proposed_index -= d;
			proposed_sign = -1;
		}
		double pvv_ = proba_v[i];

		std::vector<double> proba_v_ = getProbaVecProposal(proposed_index,proposed_sign, normal);
		double pv_v = 0;
		if (v_sign < 0)
			pv_v = proba_v_[v_index];
		else
			pv_v = proba_v_[v_index + d];

		double correction = pv_v / pvv_;
		//std::cout << "correction " << correction << "\n";

		return { proposed_index,proposed_sign, correction };
	}*/


	void JumpAtBoundaryMetropolis(Eigen::VectorXd& x,auto& n,double eps) {
		//v = -v;

		//v_sign = -v_sign;
		//return;

		double u = unif01();
		auto res = proposal(n);
		int proposed_index = std::get<0>(res);
		double proposed_sign = std::get<1>(res);
		double ratio = std::abs(n[proposed_index]) / std::abs(n[v_index]);

		if (u < ratio * std::get<2>(res)) {
			v_sign = proposed_sign;
			v_index = proposed_index;
			//std::cout << "A";// << ratio;
		}
		else {
			v_sign = -v_sign;
			//std::cout << "r";// << ratio;
		}
		
	}

	void JumpAtBoundaryBasic(Eigen::VectorXd& x, auto& n, double eps) {
		v_sign = -v_sign;
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

	Parameters tune(Polytope& K, Eigen::VectorXd x, double a, bool tuning) {

		if (!v_initialized) {
			randVelocity(x.size());
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
		int nrepeat = 500;
		double t = 0;


		while (!K.in_K(x)) {
			std::cout << x.transpose() << std::endl;
			std::cout << "BEGINING BUUUUUUUUUUUUUUUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG" << std::endl;
		}

		const auto before = std::chrono::steady_clock::now();


		Eigen::VectorXd oldx;
		for (int i = 0; i < nrepeat; i++) {
			randVelocity(x.size());
			oldx = x;
			double time = CS(K, x, eps, dim * factor);
			while (!K.in_K(x)) {
				x = oldx;
				randVelocity(x.size());
				std::cout << "NOTINK";
				if (!K.in_K(x)) {
					std::cout << " maxi bug " << i;
				}
				time = CS(K, x, eps, dim * factor);
			}
			t += time;
		}

		output_rate = 1 / (t / factor / nrepeat);
		refresh_rate = 1 / (t / factor / nrepeat);

		const sec duration = std::chrono::steady_clock::now() - before;
		//std::cout << "tune first phase:" << duration.count() << std::endl;
		const auto before2 = std::chrono::steady_clock::now();

		//std::cout << "orate " << output_rate << std::endl;

		if (tuning) {

			//std::cout << "tuning ESS" << std::endl;
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
					refresh_rate *= 1.2;
				}
				else {
					refresh_rate /= 1.2;
				}
				res = getESSPerSecInternal(K, x, a);
				new_min = std::min(res.first, res.second);
				//std::cout << " rr: " << refresh_rate << " ESS: " << new_min << " ESS2: " << getESSPerSec(K,x,a);
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

		/*std::cout << std::endl << "trials: ";
		for (int i = 0; i < 10; i++) {
			std::cout << " " << getESSPerSec(K, x, a);
		}
		std::cout << std::endl;*/

		return params;
	}

	std::pair<double, double> getESSInternal(Polytope& K, Eigen::VectorXd& x, double a) {

		int dim = x.size();
		//Eigen::VectorXd x0 = Eigen::VectorXd::Zero(dim);

		int nsamples = nSamplesESS *std::sqrt(dim);;

		for (int k = 0; k < nsamples; k++) {
			x = getNextPoint(K, x, a);
		}

		Eigen::MatrixXd samples(nsamples, dim);

		Eigen::MatrixXd samples2(nsamples, 1);

		for (int k = 0; k < nsamples; k++) {
			x = getNextPoint(K, x, a);
			for (int i = 0; i < dim; i++) {
				samples(k, i) = x(i);
			}
			samples2(k, 0) = x.norm();
		}

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

		return { min_ess / nsamples, ess2[0] / nsamples };
	}

	double getESS(Polytope& K, Eigen::VectorXd x, double a) {

		if (ESS_estimation) {
			std::cout << "NOGOOD" << std::endl;
			auto res = getESSInternal(K, x, a);
			return std::min(res.first, res.second);
		}
		else {
			std::cout << "prout" << std::endl;
			return  1. ; //1. / (std::sqrt(x.size()))
		}; //sometwhat arbitrary		
	}

	std::pair<double, double> getESSPerSecInternal(Polytope& K, Eigen::VectorXd& x, double a) {

		int dim = x.size();
		//Eigen::VectorXd x0 = Eigen::VectorXd::Zero(dim);
		

		int nsamples = nSamplesESS *std::sqrt(dim);
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

		//std::cout << "samples" << duration.count() << std::endl;
		const auto before2 = std::chrono::steady_clock::now();

		auto ess = ppl::math::ess<double>(samples);
		double min_ess = ess[0];
		for (int i = 0; i < dim; i++) {
			//std::cout << "ess i=" << i << " value = " << ess[i] / nsamples << std::endl;
			if (min_ess > ess[i]) min_ess = ess[i];
		}

		//std::cout << "ESS = " << min_ess / (double)nsamples << " for a = " << a << std::endl;

		auto ess2 = ppl::math::ess<double>(samples2);

		const sec duration2 = std::chrono::steady_clock::now() - before;
		//std::cout << "esscomput" << duration2.count() << std::endl;

		//std::cout << "ESS2 = " << ess2[0] / (double)nsamples << std::endl;

		return { min_ess / duration.count(), ess2[0] / duration.count() };
	}

	double getESSPerSec(Polytope& K, Eigen::VectorXd x, double a) {
		auto res = getESSPerSecInternal(K, x, a);
		return std::min(res.first, res.second);
	}

	double CS(Polytope& K, Eigen::VectorXd& x, double eps, int iter_max = -1) {

		int d = x.size();

		Eigen::VectorXd Ax = K.A * x;

		double t = 0;

		int nevt = 0;
		//std::cout << x.transpose() << "\n";

		while (!(iter_max > 0 && nevt > iter_max)) {

			nevt++;
			double tau_0 = getEvtTime(x);
			double tau_1 = getResampleTime();

			/*if ((Ax - K.A * x).norm() > 1e-6) {
			std::cout << "++++++++++++++++++" << "\n v=" << v.transpose() << "\n" << v_index << " " << v_sign << "\n";
				std::cout << "Ax:" << Ax << std::endl << std::endl << "true val " << K.A * x;
				std::cout << "sdf";
			}*/

			std::pair ires = get_boundary_pt(K, Ax);
			double tau_2 = ires.first;
			double tau_output = getOutputTime();
			int k = ires.second;

			//std::cout << "Ax" << Ax - K.A * x << std::endl << std::endl << K.A * x;

			if (tau_2 < 0) {
				//std::cout << "A" << K.A << std::endl << "b" << K.b << std::endl;
				std::cout << "bug, x = " << x.transpose() << " tau_2 " << tau_2 << " k " << k << " isinK " << K.in_K(x) << std::endl;
			}
			if (tau_2 - eps < 0) {
				tau_2 = eps;
				//std::cout << "PROBLEME " << tau_2 << " eps " << tau_2 - eps << " k " << k;
			}

			double tau_evt = min(tau_0, tau_1, tau_2 - eps, tau_output);  // we take tau_2 - eps to avoid going exactly on the boundary
			//std::cout << "==============" << "\n v=" << v.transpose() << "\n" << v_index << " " << v_sign << "\n";
			//std::cout << "Ax:" << Ax << std::endl << std::endl << "true val " << K.A * x;
			//std::cout << "next Ax = " << (K.A * (x + tau_evt * v)).transpose() << "\n";
			//std::cout << "update = " << (K.A * (x + tau_evt * v) - K.A * x).transpose() << "\n";
			//std::cout << "update?? = " << (K.A * (tau_evt * v) ).transpose() << "\n";
			//std::cout << "update2 = " << (tau_evt * v_sign * K.A.col(v_index)).transpose() << "\n";
			x[v_index] = x[v_index] + tau_evt * v_sign;
			t += tau_evt;
			Ax += tau_evt*v_sign* K.A.col(v_index);
			//std::cout << "v = " << v.transpose() << " v_sgn = " << v_sign << "\n";
			//std::cout << "--" << (K.A*v).transpose() << "\n" << (v_sign * K.A.col(v_index)).transpose() << "\n";
			//std::cout << "-------------" << "\n v=" << v.transpose() << "\n" << v_index << " " << v_sign << "\n";
			//std::cout << "Ax:" << Ax << std::endl << std::endl << "true val " << K.A * x;
			//std::cout << "x" << x.transpose() << "\n";
			//std::cout << "x" << x << std::endl << "v" << v << std::endl << "v_index" << v_index << std::endl << "v_sign" << v_sign << std::endl <<
			//	"Ax" << Ax << std::endl;
			nEvt++;

			if (tau_evt == tau_output) {
				break;
			}

			// work out if we are having a potential event, refreshment event, or boundary event
			if (tau_evt == tau_0) {  //potential event
				getNewVelocity(x);
				nBounce++;
			}
			else { //bounce on the boundary
				if (tau_evt == tau_1) {  //refreshment event, n.b.velocity is Gaussian
					randVelocity(d);
				}
				else {
					nReflections++;
					auto normal = K.A.row(k); // pull normal from
					//JumpAtBoundaryBasic(x, normal, eps);
					JumpAtBoundaryMetropolis(x, normal, eps);
					//std::cout << "reflection" << x.transpose() << "\n";
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

		Eigen::VectorXd oldx = x;
		if (!v_initialized) {
			randVelocity(x.size());
			v_initialized = true;
		}
		sigma = std::sqrt(1. / 2. / a);
		CS(K, x, eps);

		int niter = 0;
		while (!K.in_K(x)) {
			if (!K.in_K(oldx)) {
				std::cout << "big bug " << std::endl;
			}
			niter++;
			std::cout << "not in K!!" << niter;
			//std::cout << "x " << x.transpose() << std::endl << "xold " << oldx.transpose() << std::endl;
			x = oldx;
			randVelocity(x.size());
			CS(K, x, eps);
		}

		//std::cout << x.transpose() << std::endl;
		return x;
	}

	double eps = 1e-6;
	double sigma = 1;
	//Eigen::VectorXd v;
	int v_index = 0;
	double v_sign = 1;
	double refresh_rate = 0.1;
	double output_rate = 0.5;

	std::default_random_engine rng;

	int nBounce = 0;
	int nReflections = 0;
	int nEvt = 0;

	int nSamplesESS = 100;

	bool ESS_estimation = true;

	double reflection_parameter = 10;

	bool v_initialized;
};