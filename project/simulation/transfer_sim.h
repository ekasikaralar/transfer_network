#ifndef TRANSFER_SIM_H
#define TRANSFER_SIM_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "gurobi_c++.h"
#include <nlohmann/json.hpp>
#include "nn_policy.h"

namespace simulation
{
	class Simulation
	{
		public:
			Simulation(const std::string& jsonFileName);
			~Simulation();

			int save(int start_iter, int end_iter, const std::string& record_file);

		private:
			std::vector<std::string> splitString(const std::string& input, char delimiter);
			std::vector<std::vector<double>> readMatrixFromCSV(const std::string& filename);
			std::vector<double> readVectorFromCSV(const std::string& filename);

			int class_no;          // I, original and rework classes
			int station_no;        // K, server pools
			int num_interval;      // M, grid intervals on the horizon
			int num_iterations;    // default replication count
			int profile_iters;     // replications whose interval profile is written
			double interval_len;   // length of one interval, in hours
			std::string policy;    // cmu, cmu_theta, fsf, myopic, myopic_theta, value, nn, prices
			NNPolicy nn;           // the trained policy or the price table, loaded when policy is nn or prices

			// time varying, one row per interval
			std::vector<double> lambda;                       // total arrival rate, per hour
			std::vector<std::vector<int>> no_server;          // interval x pool
			std::vector<std::vector<double>> arr_cdf;         // interval x class
			std::vector<std::vector<double>> optimal_xi;      // interval x (class*pool)

			// static
			std::vector<std::vector<double>> mu_hourly;       // class x pool, zero off the edges
			std::vector<double> theta_hourly;                 // abandonment rate per class
			std::vector<double> cost_rate;                    // c_i = h_i + theta_i a_i, for the weights
			std::vector<double> holding_cost;                 // h_i, per caller in queue per hour
			std::vector<double> abandon_penalty;              // a_i, per abandonment
			std::vector<std::vector<double>> edges;           // class x pool, 1 on an activity

			// transfers and values
			std::vector<std::vector<double>> routing_P;       // (class*pool) x class
			std::vector<std::vector<double>> v_activity;      // class x pool, v_j
			std::vector<std::vector<double>> vbar_activity;   // class x pool, vbar_j
			std::vector<double> kappa_class;                  // kappa_l by destination class
	};

	class Execute
	{
		public:
			Execute(int class_no_, int station_no_, int num_interval_,
				const std::vector<std::vector<double>>& arr_cdf_,
				const std::vector<std::vector<double>>& mu_hourly_,
				const std::vector<double>& theta_hourly_,
				const std::vector<std::vector<int>>& no_server_,
				const std::vector<double>& cost_rate_,
				const std::vector<double>& holding_cost_,
				const std::vector<double>& abandon_penalty_,
				const std::vector<std::vector<double>>& optimal_xi_,
				const std::vector<std::vector<double>>& edges_,
				const std::vector<double>& lambda_,
				const std::vector<std::vector<double>>& routing_P_,
				const std::vector<std::vector<double>>& v_activity_,
				const std::vector<std::vector<double>>& vbar_activity_,
				const std::vector<double>& kappa_class_,
				double interval_len_,
				const std::string& policy_,
				const NNPolicy* nn_,
				int seed);
			~Execute();

			// One replication.  Returns the row written to the results file.
			std::vector<double> run();
			void save_interval_profile(const std::string& filename);

		private:
			// --- assignment linear program (Gurobi) ---
			GRBEnv env = GRBEnv();
			GRBModel model = GRBModel(env);
			std::vector<std::vector<GRBVar>> d;              // class x pool
			std::vector<GRBConstr> class_constraints;
			std::vector<GRBConstr> station_constraints;
			std::vector<std::vector<double>> weight;          // w_j of the rule, class x pool
			const NNPolicy* nn;                               // null unless policy is nn
			void refresh_weights();                           // nn: weights at the current state
			std::vector<std::vector<double>> optimal_d;

			void build_model();
			const std::vector<std::vector<double>>& linear_program();
			void refresh_station_capacity();

			// --- events ---
			void queue_init();
			double generate_interarrival();
			double generate_abandon(int cls);
			double generate_service(int station);
			void add_queue(double arr_time, int cls);
			void remove_queue(int cls);
			void handle_arrival_event(int cls);
			void handle_depart_event(int station, int cls);
			void handle_abandon_event();
			void system_arrangement();
			void rearrange_system();
			int choose_station_for_arrival(int cls);

			// --- time varying primitives ---
			int  interval_of(double t) const;
			void load_interval(int iv);

			// --- transfers ---
			int  draw_destination(int cls, int station);   // -1 means the caller exits
			void inject_transfer(int dest_cls);

			// --- clock and event times ---
			double sim_clock;
			double t_arrival;
			double t_abandon;
			double t_event;
			std::vector<double> t_depart;                    // one per pool
			int class_abandon;
			int cust_abandon;
			std::vector<double> next_abandon_time;
			static constexpr double inf = std::numeric_limits<double>::infinity();

			// --- state ---
			// num_in_system[0] is the total, num_in_system[i+1] class i.
			// num_in_service[k+1][i+1] is class i in pool k, with row 0 and
			// column 0 holding the totals.
			std::vector<int> num_in_system;
			std::vector<std::vector<int>> num_in_service;
			std::vector<int> num_in_queue;
			std::vector<std::deque<double>> queue_list;          // arrival times, per class
			std::vector<std::vector<double>> abandonment_list;   // abandonment times, per class
			std::vector<int> num_served_buf;
			std::vector<int> optimal_queue_len_buf;
			std::vector<double> numerator_tmp, prob_tmp, ser_cdf_tmp;

			// --- counters of the replication ---
			std::vector<int> num_arrivals, num_abandons, num_departures, num_transfers, num_resolved;
			std::vector<double> queue_integral, system_integral;
			double value_resolved;      // sum of v_j over resolving completions
			double penalty_transfers;   // sum of kappa_l over transfers
			double cost_holding;        // sum_i h_i int Y_i
			double cost_abandon;        // sum_i a_i A_i(T)
			double value_integral;      // int sum_j vbar_j mu_j psi_j(s) ds, the value term of the paper's objective

			// per interval profile
			std::vector<double> iv_time, iv_queue, iv_service, iv_servers, iv_arrivals, iv_abandons, iv_value;

			// --- primitives ---
			const int class_no;
			const int station_no;
			const int num_interval;
			const double interval_len;
			const std::vector<std::vector<double>> mu_hourly;
			const std::vector<double> theta_hourly;
			const std::vector<double> cost_rate;
			const std::vector<double> holding_cost;
			const std::vector<double> abandon_penalty;
			const std::vector<std::vector<double>> edges;
			const std::vector<std::vector<double>> arr_cdf_all;
			const std::vector<std::vector<int>> no_server_all;
			const std::vector<std::vector<double>> optimal_xi_all;
			const std::vector<double> lambda_all;
			const std::vector<std::vector<double>> routing_P;
			const std::vector<std::vector<double>> v_activity;
			const std::vector<std::vector<double>> vbar_activity;
			const std::vector<double> kappa_class;
			const std::string policy;

			// current interval
			int interval;
			std::vector<int> no_server;
			std::vector<double> arr_cdf;
			int total_servers;

			// --- random numbers ---
			std::mt19937_64 generator;
			std::uniform_real_distribution<double> uniform01;
			std::exponential_distribution<double> interarrivalDist;
			std::vector<std::exponential_distribution<double>> abandonDist;
	};
}

#endif
