// FigureESS.hpp
//
#define EIGEN_DONT_PARALLELIZE
#include "VolumeComputation.h"

#include <concepts>
#include <chrono>
#include <tuple>

#include "ESS.hpp"
#include "BPS_Partial_CGAL.hpp"

#include<fstream>

//using clock = std::chrono::system_clock;
using sec = std::chrono::duration<double>;


enum ESSType {
	ESS_PER_SAMPLE,
	ESS_PER_SECOND
};

enum TunningParameter {
	REFRESH_RATE,
	OUTPUT_RATE
};

class Analysis1D {
public:

	Analysis1D(GaussianSamplerBPSPartialCGAL sampler, Polytope K_) : samplerBPS(sampler), K(K_) {};

	GaussianSamplerBPSPartialCGAL samplerBPS;
	Polytope K; 
	int dim; 
	double a;
	std::vector<double> factors; 
	ESSType ess_type; 
	TunningParameter parameter;
	std::string outputFile;
	int num_repeats;
};

void analyseTunning1D(Analysis1D params) {

	double base_out = params.samplerBPS.output_rate;
	double base_refresh = params.samplerBPS.refresh_rate;
	double base_nsamples = params.samplerBPS.nSamplesESS;


	std::vector<std::vector<double>> results(params.factors.size());
	std::vector<double> rates(params.factors.size());

	for (int i = 0; i < params.factors.size(); i++) {
		results[i].resize(params.num_repeats);
		//results[i] = 0;
		for (int k = 0; k < params.num_repeats; k++) {
			if (params.parameter == OUTPUT_RATE) {
				params.samplerBPS.output_rate = base_out * params.factors[i];
				params.samplerBPS.nSamplesESS = std::max({ base_nsamples,base_nsamples * params.factors[i] });
				rates[i] = params.samplerBPS.output_rate;
			}
			else {
				params.samplerBPS.refresh_rate = base_refresh * params.factors[i];
				rates[i] = params.samplerBPS.refresh_rate;
			}
			
			Eigen::VectorXd x = Eigen::VectorXd::Zero(params.dim);

			if(params.ess_type == ESS_PER_SECOND)
				results[i][k] = params.samplerBPS.getESSPerSec(params.K, x, params.a);
			else {
				auto res = params.samplerBPS.getESS(params.K, x, params.a);
				results[i][k] = std::get<0>(res) / std::get<1>(res);
			}
		}
		//std::cout << "ESS per sec " << results[i][k] << 
		//	" refresh " << params.samplerBPS.refresh_rate << " output " << params.samplerBPS.output_rate << std::endl;
	}

	std::ofstream out;
	out.open(params.outputFile, std::ios::out);

	for (int i = 0; i < results.size(); i++) {
		for(int j = 0; j < results[i].size(); j++)
			out << results[i][j] << " ";
		out << std::endl;
	}

	out.close();

	//std::ofstream out;
	out.open("rates_" + params.outputFile, std::ios::out);

	for (int i = 0; i < params.factors.size(); i++) {
		out << rates[i] << " ";
	}

	out.close();

}


void analyseTunning() {


	//double a = 0.33;
	double a = 2.7;//  1. / 100000000000000000;
	//double a = 10000000000000000000000;
	int dim = 100;
	std::pair<Polytope, MP_FLOAT> resPoly = createPolytope("cube", dim);
	Polytope K = resPoly.first;
	//K.compute_An();
	GaussianSamplerBPSPartialCGAL samplerBPS;
	samplerBPS.nSamplesESS = 10000;
	Eigen::VectorXd x = Eigen::VectorXd::Zero(dim);
	samplerBPS.tune(K, x, a, true);

	samplerBPS.nSamplesESS = 10000;

	double base_out = samplerBPS.output_rate;
	double base_refresh = samplerBPS.refresh_rate;
	double base_nsamples = samplerBPS.nSamplesESS;
	samplerBPS.nSamplesESS = base_nsamples;
	std::vector<double> factors;
	factors.push_back(1.);
	for (int i = 1; i < 5; i++) {
		factors.push_back(std::pow(2, i));
		factors.insert(factors.begin(), 1. / std::pow(2, i));
		//factors[i] = std::pow(2,i);
		//factors[5+i] = 1./std::pow(2,i);
	}

	Analysis1D params(samplerBPS,K);
	params.a = a;
	params.dim = dim;
	params.ess_type = ESS_PER_SECOND;
	params.parameter = OUTPUT_RATE;
	params.factors = factors;
	params.outputFile = "output_tunning_analysis_cube" + std::to_string(dim) + ".txt";
	params.num_repeats = 10;
	//analyseTunning1D(params);

	//resest the samplerBPS:
	params.samplerBPS = samplerBPS;
	params.ess_type = ESS_PER_SAMPLE;
	params.outputFile = "per_sample_output_tunning_analysis_cube" + std::to_string(dim) + ".txt";
	//analyseTunning1D(params);

	//resest the samplerBPS:
	params.samplerBPS = samplerBPS;
	params.ess_type = ESS_PER_SAMPLE;
	params.parameter = REFRESH_RATE;
	params.outputFile = "per_sample_refresh_tunning_analysis_cube" + std::to_string(dim) + ".txt";
	analyseTunning1D(params);

	//resest the samplerBPS:
	params.samplerBPS = samplerBPS;
	params.ess_type = ESS_PER_SECOND;
	params.parameter = REFRESH_RATE;
	params.outputFile = "refresh_tunning_analysis_cube" + std::to_string(dim) + ".txt";
	analyseTunning1D(params);
	return;

	
}


int main()
{
	analyseTunning();
	return -1;

}
