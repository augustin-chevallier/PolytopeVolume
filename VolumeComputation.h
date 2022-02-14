// VolumeComputation.h : fichier Include pour les fichiers Include système standard,
// ou les fichiers Include spécifiques aux projets.

#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <numeric>
#include "Polytope.hpp"
#include "AnnealingSchedule.hpp"
#include <omp.h>
#include <chrono>
// TODO: Référencez ici les en-têtes supplémentaires nécessaires à votre programme.

using sec = std::chrono::duration<double>;


struct VolumeResult {
    MP_FLOAT volume;
    double T;
    uint64_t steps;
    uint64_t num_oracle;
    uint64_t num_exits;
    uint64_t num_exits_resample;
    double tunning_time;
};

template<class GaussianSampler>
VolumeResult volumeEstimation(Polytope P, uint64_t budget, bool tunning, bool ESS, int num_threads){

    omp_set_num_threads(num_threads);


    std::cout << "===========================\n";
    std::cout << "Volume computation, dim = " << P.dim << std::endl;      


    std::vector<GaussianSampler> sampler(num_threads);

    for (int i = 0; i < sampler.size(); i++) {
        sampler[i].ESS_estimation = ESS;
    }


    //TODO: eps as a param?
    double eps = 0.01;

    const auto before = std::chrono::steady_clock::now();


    std::cout << "Computing annealing schedule and tuning random walk" << std::endl;
    auto res0 = getAnnealingSchedule<GaussianSampler>(P,sampler,tunning);
    double initialVolumeRatio = std::get<2>(res0);
    std::cout << "Annealing schedule and tuning completed" << std::endl;
    const sec durationAnnealing = std::chrono::steady_clock::now() - before;
    const auto before2 = std::chrono::steady_clock::now();

    std::vector<double> annealing_schedule = std::get<0>(res0);
    //eps = res0.second;
    std::vector<std::pair<typename GaussianSampler::Parameters,double>> paramsVec = std::get<1>(res0);
    int m = annealing_schedule.size();


    std::vector<Eigen::VectorXd> x(num_threads);
    for (int j = 0; j < num_threads; j++) {
        x[j] = Eigen::VectorXd::Zero(P.dim);
    }


    std::cout << "initial volume ratio: " << initialVolumeRatio << std::endl;
    MP_FLOAT volume = boost::multiprecision::pow( MP_FLOAT(M_PI) / annealing_schedule[0] , P.dim / 2. )*initialVolumeRatio;

//    std::vector<double> functions_ratio(annealing_schedule.size(),0);
//    std::vector<double> samples_number(annealing_schedule.size(), 0);

    std::cout << "Number of  phases: " << m << "\n";


    int num_phases = annealing_schedule.size() - 1;


    std::vector<uint64_t> num_steps(annealing_schedule.size() - 1);
    double sum_inverse_ess = 0;
    for (int i = 0; i < annealing_schedule.size() - 1;i++) {
        double ess = paramsVec[i].second;
        sum_inverse_ess += 1. / ess;
    }
    for (int i = 0; i < annealing_schedule.size() - 1;i++) {
        double alpha = budget / sum_inverse_ess;
        double ess = paramsVec[i].second;
        num_steps[i] = alpha / ess;
        //num_steps[i] = budget / a_sched.size();
    }

    double total_steps = 0;

    for (int i = 0; i < annealing_schedule.size() - 1; i++) {

        double functions_ratio = 0;
        double samples_number = 0;

        std::cout << "Starting phase " << i << ", current sigma: " << std::sqrt(1./2./annealing_schedule[i]) 
            << ", current volume: " << volume << std::endl;
        
        
        for (int j = 0; j < num_threads; j++) {
            sampler[j].setParameters(paramsVec[i].first);
        }
        double ess = paramsVec[i].second;

        //int num_steps = 100. / ess  / batch_size / 0.002;
        std::cout << "num steps = " << num_steps[i] << std::endl;

        #pragma omp parallel for
        for (int j = 0; j < num_threads; j++) {
            double acc = 0;
            for (int k = 0; k < num_steps[i] / num_threads; k++) {
                x[j] = sampler[j].getNextPoint(P, x[j], annealing_schedule[i]);

                if (!P.in_K(x[j])) {
                    std::cerr << "Bug: a point was not in K." << std::endl;
                }
                acc += gaussian_function(x[j], annealing_schedule[i + 1]) / gaussian_function(x[j], annealing_schedule[i]);
            }
            #pragma omp atomic
            samples_number += num_steps[i] / num_threads;
            #pragma omp atomic
            functions_ratio += acc;
        }


        std::cout << "Final number of steps in phase " << i << ": " << samples_number <<std::endl;
        
        volume = volume * functions_ratio / samples_number;
        total_steps += samples_number;
    }


    const sec durationComputation = std::chrono::steady_clock::now() - before;

    std::cout << "\n\n" << "duration annealing: " << durationAnnealing.count() << "\nduration volume: " << durationComputation.count() << std::endl;

    VolumeResult res;
    res.tunning_time = durationAnnealing.count();
    res.steps = total_steps;
    res.volume = volume;
    res.num_oracle = 0;
    res.num_exits = 0;
    res.num_exits_resample = 0;
    for (int j = 0; j < num_threads; j++) {
        res.num_oracle += sampler[j].nOracles;
        res.num_exits += sampler[j].num_exit;
        res.num_exits_resample += sampler[j].num_exit_resample;
    }
    return res;
}


