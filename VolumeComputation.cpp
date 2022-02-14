// VolumeComputation.cpp : définit le point d'entrée de l'application.
//
//#define EIGEN_USE_BLAS
#define EIGEN_DONT_PARALLELIZE

//#define EIGEN_USE_BLAS
//export OPENBLAS_NUM_THREADS = 4

#define _HAS_DEPRECATED_RESULT_OF true

//#define USE_GPU


#include "VolumeComputation.h"

#include <concepts>
#include <chrono>
#include <fstream>

#include "BPS_Partial_CGAL.hpp"

#include <boost/program_options.hpp>

using sec = std::chrono::duration<double>;

namespace po = boost::program_options;


//extern "C" void openblas_set_num_threads(int num_threads);


int main(int argc, const char* argv[])
{

	//cublasHandle_t handle;
	/*cublasCreate(&handle);
	int d = 1000;
	std::cout << "cuda started" << std::endl;
	std::pair<Polytope, MP_FLOAT> resPoly1 = createPolytope("cube", d, false);
	Eigen::MatrixXd A = resPoly1.first.A;
	Eigen::VectorXd x = Eigen::VectorXd::Ones(d);
	Eigen::VectorXd b = Eigen::VectorXd::Ones(2 * d);

	std::cout << "starting computations " << std::endl;

	int nr_rows_A = 2 * d;
	int nr_cols_A = d;
	CUDA_FT* d_A, * d_x, *d_b;
	int* d_i;
	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(CUDA_FT));
	cudaMalloc(&d_x, d * sizeof(CUDA_FT));
	cudaMalloc(&d_b, nr_rows_A * sizeof(CUDA_FT));
	cudaMalloc(&d_i, 1 * sizeof(int));

	copyVecToDevice(x, d_x);
	copyMatToDevice(A, d_A);
	copyVecToDevice(b, d_b);

	const auto before1 = std::chrono::steady_clock::now();
	//std::cout << A.cols() << " " << A.rows() <<  " " << x.size() << std::endl;

	for (int i = 0; i < 1000; i++) {
		b = b + A * x;
		double qsd = b.maxCoeff();
		b = b * qsd;
	}
	const sec duration1 = std::chrono::steady_clock::now() - before1;

	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	CUDA_FT* cst_one;
	cudaMalloc(&cst_one, 1 * sizeof(CUDA_FT));

	std::cout << "duration " << duration1.count() << std::endl;
	//x = Eigen::VectorXd::Zero(d);
	//b = Eigen::VectorXd::Zero(2 * d);
	const auto before2 = std::chrono::steady_clock::now();
	const CUDA_FT alpha = 1;
	const int lda = nr_rows_A;
	const CUDA_FT beta = 1;
	std::vector<CUDA_FT> buf(b.size());
	for (int i = 0; i < 1000; i++) {
		cublasDgemv(handle, CUBLAS_OP_N, nr_rows_A, nr_cols_A, cst_one, d_A, lda, d_x, 1, cst_one, d_b, 1);
		//cudaMemcpy(buf.data(), d_b, nr_rows_A * sizeof(CUDA_FT), cudaMemcpyDeviceToHost);
		//auto vec = VecFromDevice(d_b, nr_rows_A);
		//b = vec.maxCoeff() * b;
		//testThrust(handle,d_b,nr_rows_A,d_i);
	}
	cudaDeviceSynchronize();
	const sec duration2 = std::chrono::steady_clock::now() - before2;

	Eigen::VectorXd b_ = VecFromDevice(d_b, b.size());
	Eigen::MatrixXd A_ = MatFromDevice(d_A, nr_cols_A, nr_rows_A);
	std::cout << "duration bis " << duration2.count() << std::endl;
	std::cout << (b- b_).norm() << "\n" << (A_ - A).norm();

	cudaFree(d_b);
	cudaFree(d_A);
	cudaFree(d_x);
	cublasDestroy(handle);

	return 1;*/


	std::string polytope;
	std::string A_file;
	std::string b_file;
	std::string pt0_file;
	int dim = 50;
	uint64_t budget = 1e6;
	int num_threads = 1;
	std::string output;
	int redirect_output = 0;
	int tuning = 1;
	int ESS = 1;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("polytope,p", po::value<std::string>(&polytope)->default_value("file"), "possible polytopes: cube,standard_simplex, isotropic_simplex, file (reading A.txt and b.txt)")
		("dim,d", po::value<int>(&dim)->default_value(50), "dimension of the polytope")
		("budget,n", po::value<uint64_t>(&budget)->default_value(100000), "number of steps (excluding tunning)")
		("tuning,t", po::value<int>(&tuning)->default_value(1), "tuning of the random walk (may be expensive)")
		("ESS,e", po::value<int>(&ESS)->default_value(1), "tuning of the random walk (may be expensive)")
		("num_threads", po::value<int>(&num_threads)->default_value(1), "compute an ESS approximation or not (may be expensive)")
		("output,o", po::value<std::string>(&output)->default_value("out"), "output file, without the extension")
		("redirect_output,l", po::value<int>(&redirect_output)->default_value(0), "redirect output to file (0 or 1)")
		("A_file,A", po::value<std::string>(&A_file)->default_value("A.txt"), "file with the hyperplan matrix A (if using the file polytope)")
		("b_file,b", po::value<std::string>(&b_file)->default_value("b.txt"), "file with the hyperplan vector b (if using the file polytope)")
		("pt0_file", po::value<std::string>(&pt0_file)->default_value(""), "file with a point inside the convex. If none is given, 0 is assumed to be in the polytope.")
		;

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).
		options(desc).run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << "Usage: options_description [options]\n";
		std::cout << desc;
		return 0;
	}

	//openblas_set_num_threads(num_threads);
	//num_threads = 1;


	std::ofstream outLog;
	std::streambuf* coutbuf;
	if (redirect_output == 1) {
		outLog = std::ofstream(output + ".log");
		coutbuf = std::cout.rdbuf(); 
		std::cout.rdbuf(outLog.rdbuf()); 
	}

	Eigen::initParallel();
	
		std::pair<Polytope, MP_FLOAT> resPoly = createPolytope(polytope, dim, false,A_file,b_file);

		Polytope K = resPoly.first;
		MP_FLOAT true_vol = resPoly.second;

		//if we have a initial point, we translate by pt0 the polytope Ax < b to A(x+pt0) < b to bring pt0 to 0
		if (pt0_file != "") {
			std::vector<double> pt0_;
			std::ifstream file;
			file.open(pt0_file);
			std::string str;
			while (std::getline(file, str)) {
				std::istringstream sin(str);
				double tmp;
				while (sin >> tmp) {
					pt0_.push_back(tmp);
				}
			}

			Eigen::VectorXd pt0;
			pt0.resize(pt0_.size());
			for (int i = 0; i < pt0_.size(); i++) {
				pt0[i] = pt0_[i];
			}

			K.b = K.b - K.A * pt0;
		}

		std::cout << "true vol " << true_vol << std::endl;
		//return -1;

		const auto before = std::chrono::steady_clock::now();
		auto res = volumeEstimation<GaussianSamplerBPSPartialCGAL>(K, budget, tuning, ESS, num_threads);
		const sec duration = std::chrono::steady_clock::now() - before;


		std::cout << "\nIt took " << duration.count() << "s" << std::endl;

		std::cout << "\nVolume :" << res.volume <<
			"\n erreur : " << boost::multiprecision::abs(res.volume - true_vol) / true_vol <<
			"\n log error: " << boost::multiprecision::log(res.volume) - boost::multiprecision::log(true_vol) << std::endl;
		std::cout << "true volume " << true_vol << std::endl;

		std::ofstream out;
		out.open(output + ".txt", std::ios::out);
		out << "vol_estimate " << res.volume << std::endl;
		out << "vol_exact " << true_vol << std::endl;
		out << "num_samples " << res.steps << std::endl;
		out << "num_oracles " << res.num_oracle << std::endl;
		out << "num_exits " << res.num_exits << std::endl;
		out << "num_exits_resample " << res.num_exits_resample << std::endl;
		out << "time " << duration.count() << std::endl;
		out << "time_annealing " << res.tunning_time << std::endl;
		out << "tuning_rw " << tuning << std::endl;
		out << "parameters: dim = " << dim << ", polytope = " << polytope << ", budget = " << budget;

		out.close();

	if (redirect_output == 1) {
		std::cout.rdbuf(coutbuf);
		outLog.close();
	}


	return 0;
}
