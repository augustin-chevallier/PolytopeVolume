#pragma once
#include <iostream>
#include <tuple>
#include "Polytope.hpp"
#include "Utilities.hpp"

#include <random>

template <class GaussianSampler>
std::pair<double,double> getInnerGaussian(Polytope& K, GaussianSampler& sampler) {

    // compute the set of distances to each hyperplane
    std::vector<double> distances(K.A.rows());
    for (int i = 0; i < distances.size(); i++) {
        distances[i] = K.b(i) / K.A.row(i).norm();
    }

    double initial_a = 1;


    //first we try to get a reasonable value from a, probably overestimated (see Cousins et Al paper)
    int iterations = 0;
    while (iterations < 1e4) {
        iterations = iterations + 1;
        double sum = 0;
        for (int i = 0; i < distances.size(); i++) {
            sum = sum + std::exp(-initial_a * distances[i] * distances[i]) / (2 * distances[i] * std::sqrt(M_PI * initial_a));
        }

        //this number here is entierly arbitrary
        if (sum > 1) {
            initial_a = initial_a * 10;
        }
        else {
            break;
        }
    }


    //Then we optimize this value
    //ratio of points sampled from a given gaussian that are also in K
    double ratio = 0;

    int dim = K.dim;
    Eigen::VectorXd x(dim);

    std::random_device rd;
    auto rng = std::default_random_engine(rd());

    double a = initial_a/10;

    double modification_factor = 10;
    bool last_up = false;
    bool first_step = true;
    //target a ratio between 0.1 and 0.2
    do{

        if (!first_step) {
            if (ratio < 0.1) {
                if (last_up) {
                    modification_factor = std::sqrt(modification_factor);
                }
                a *= modification_factor;
                last_up = false;
            }
            else {
                if (!last_up) {
                    modification_factor = std::sqrt(modification_factor);
                }
                a /= modification_factor;
                last_up = true;
            }
        }

        double sigma = std::sqrt(1. / 2. / a);

        int num_trials = 1000;
        int num_in_K = 0;
        for (int k = 0; k < num_trials; k++) {
            for (int i = 0; i < dim; i++) {
                x[i] = std::normal_distribution<double>(0, sigma)(rng);
            }
            num_in_K += K.in_K(x);
        }
        ratio = (double)num_in_K / (double)num_trials;
        
        first_step = false;
    } while (ratio < 0.1 || ratio > 0.2);

    //do a final computation of the ratio with more points this time
    double sigma = std::sqrt(1. / 2. / a);

    int num_trials = 10000;
    int num_in_K = 0;
    for (int k = 0; k < num_trials; k++) {
        for (int i = 0; i < dim; i++) {
            x[i] = std::normal_distribution<double>(0, sigma)(rng);
        }
        num_in_K += K.in_K(x);
    }
    ratio = (double)num_in_K / (double)num_trials;

    return { a,ratio };
}


template <class GaussianSampler>
std::tuple<double,Eigen::VectorXd,typename GaussianSampler::Parameters,double> getFollowingGaussian(double a, int target_independant_samples, Polytope K, Eigen::VectorXd x, GaussianSampler& sampler,bool tunning){

    //see lemma 3 from Cousins et Al paper, A practical volume algorithm
    double a_ratio = 1 - 1. / (double)K.dim;

    double last_a = a;

    bool done = false;
    int r = 1; //see lemma 3

    auto params = sampler.tune(K, x, last_a,tunning);
    std::tuple<double, int,int,double> essRes = sampler.getESS(K, x, last_a);
    double ess = std::get<0>(essRes) / std::get<1>(essRes);
    int samples_number = target_independant_samples / ess;


    //first thing, sample some points according to the current Gaussian
    std::vector<Eigen::VectorXd> pts(samples_number);
    for (int i = 0; i < samples_number; i++) {
        x = sampler.getNextPoint(K, x, last_a);
        pts[i] = x;
    }


    double last_ratio = 0.1;
    std::vector<double> functions_ratio(samples_number, 0);

    while (!done){
        a = last_a * std::pow(a_ratio,r);
        
        //std::cout << "proposed a = " << a << " last_a = " << last_a << std::endl;

        for (int i = 0; i < samples_number; i++) {
            functions_ratio[i] = gaussian_function(pts[i], a) / gaussian_function(pts[i], last_a);
        }
 

        if (var(functions_ratio) / (std::pow(mean(functions_ratio) ,2) ) >= 2 || mean(functions_ratio) / last_ratio < 1 + 1e-5) {
            if (r != 1) {
                r = r / 2;
            }
            done = 1;
        }
        else{
            r = 2 * r;
        }
        last_ratio = mean(functions_ratio);
    }

    double new_a = last_a * std::pow(a_ratio,r);

    return { new_a,x, params, ess };
}

template<class GaussianSampler>
std::tuple<std::vector<double>, std::vector<std::pair<typename GaussianSampler::Parameters,double>>,double> getAnnealingSchedule(Polytope K, std::vector<GaussianSampler>& samplers,bool tunning) {

    int num_threads = samplers.size();

    std::vector<double> annealing_schedule;

    //get the first gaussian
    auto res0 = getInnerGaussian(K, samplers[0]);
    double volumeRatio = res0.second;
    annealing_schedule.push_back(res0.first);

    std::vector<std::pair<typename GaussianSampler::Parameters,double>> paramsMap;
    std::vector<Eigen::VectorXd> x(num_threads,Eigen::VectorXd::Zero(K.dim));

    int it = 0;

    double ratio = 2;

    while (ratio > 1.001) {
        it++;

        int n_independant_samples = 100;
        auto res = getFollowingGaussian(annealing_schedule[it - 1], n_independant_samples, K, x[0], samplers[0],tunning);
        //double next_a = std::get<0>(res);
        //annealing_schedule.push_back(std::get<0>(res));

        //get the tunning results
        paramsMap.push_back({ std::get<2>(res),std::get<3>(res) });
        for (int j = 0; j < num_threads; j++) {
            x[j] = std::get<1>(res);
            samplers[j].setParameters(std::get<2>(res));
        }

        double next_a = std::get<0>(res);

        double ess = std::get<3>(res);
        
        double functions_ratio = 0;
        double n_samples = 0;

        int num_pts = 100 / ess;// / num_threads;
        #pragma omp parallel for
        for (int j = 0; j < num_threads; j++) {
            double acc_functions = 0;
            int acc_n_samples = 0;
            for (int k = 0; k < num_pts; k++) {

                x[j] = samplers[j].getNextPoint(K, x[j], annealing_schedule[it - 1]);

                acc_n_samples ++;
                acc_functions += gaussian_function(x[j], next_a) / gaussian_function(x[j], annealing_schedule[it - 1]);
            }
            #pragma omp atomic
            n_samples += acc_n_samples;
            #pragma omp atomic
            functions_ratio += acc_functions;
        }
        ratio = functions_ratio / n_samples;

        if (ratio > 1.001) {
            annealing_schedule.push_back(next_a);
        }
        else {
            annealing_schedule.pop_back();
            annealing_schedule.push_back(0);
        }
    }
    /*if (annealing_schedule[it] >= 0) {
        annealing_schedule.pop_back();
        annealing_schedule[annealing_schedule.size() - 1] = 0;
    }
    else {
        annealing_schedule[annealing_schedule.size() - 1] = 0;
    }*/

    std::cout << "\n\n================== \n\nANNEALING SCHEDULE:\n";
    for (int i = 0; i < annealing_schedule.size(); i++) {
        std::cout << annealing_schedule[i] << std::endl;
    }
    std::cout << "=========================\n\n";


    return {annealing_schedule, paramsMap, volumeRatio};
}