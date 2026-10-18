#ifndef INDEX_SIM_H
#define INDEX_SIM_H

// Non preemptive simulation of the transfer network under index rules, with
// no linear program.  Same inputs, same rules and same objective as
// transfer_sim, but a caller in service stays with the agent until the
// service ends, and assignments are made one at a time at the events:
//
//   arrival      the caller goes to the idle pool that can serve the class
//                and has the largest weight w_ij, or waits in queue
//   departure    the freed agent takes, among the waiting classes its pool
//                can serve, the one with the largest weight w_ij, first come
//                first served within the class
//   abandonment  the caller leaves the queue
//   staffing     at an interval boundary, agents joining a pool take waiting
//                callers by the same rule; agents leaving finish their
//                service first
//
// Objective, as in transfer_sim: value of resolving completions minus
// transfer penalties minus holding cost minus abandonment penalties.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <nlohmann/json.hpp>
#include "nn_policy.h"

namespace simulation
{
	struct Inputs {
		int class_no, station_no, num_interval, num_iterations, profile_iters;
		double interval_len;
		std::string policy;
		std::vector<double> lambda;
		std::vector<std::vector<int>> no_server;
		std::vector<std::vector<double>> arr_cdf, optimal_xi, mu_hourly, routing_P, v_activity, vbar_activity;
		std::vector<double> theta_hourly, cost_rate, holding_cost, abandon_penalty, kappa_class;
		NNPolicy nn;                                 // the trained policy or the price table, loaded when policy is nn or prices
	};

	Inputs read_inputs(const std::string& jsonFileName);
	int run_all(const Inputs& in, int start_iter, int end_iter, const std::string& record_file);

	class Execute
	{
		public:
			Execute(const Inputs& in, int seed);
			std::vector<double> run();
			void save_interval_profile(const std::string& filename);
			const std::vector<int>& class_maxima() const { return max_in_system; }

		private:
			const Inputs& in;
			std::vector<std::vector<double>> weight;    // w_ij of the rule, class x pool
			void refresh_weights();                     // nn: weights at the current state

			// events
			void handle_arrival_event(int cls);
			void handle_depart_event(int station, int cls);
			void handle_abandon_event();
			void assign_from_queue(int station);        // one freed agent of the pool
			void fill_idle_agents();                    // after a staffing change
			void start_service(int cls, int station);
			void add_queue(double arr_time, int cls);
			void remove_queue(int cls);
			void refresh_abandon_clock();
			int  interval_of(double t) const;
			void load_interval(int iv);
			int  draw_destination(int cls, int station);
			double generate_service(int station);

			// clock and event times
			double sim_clock, t_arrival, t_abandon, t_event;
			std::vector<double> t_depart;
			int class_abandon, cust_abandon;
			std::vector<double> next_abandon_time;
			static constexpr double inf = std::numeric_limits<double>::infinity();

			// state, index 0 holds the totals
			std::vector<int> num_in_system, num_in_queue;
			std::vector<std::vector<int>> num_in_service;      // [pool+1][class+1]
			std::vector<std::deque<double>> queue_list;
			std::vector<std::vector<double>> abandonment_list;

			// counters
			std::vector<int> num_arrivals, num_abandons, num_departures, num_transfers;
			std::vector<double> queue_integral, system_integral;
			std::vector<int> max_in_system;             // per class, over the replication
			double value_resolved, penalty_transfers, cost_holding, cost_abandon, value_integral;
			std::vector<double> iv_time, iv_queue, iv_service, iv_servers, iv_arrivals, iv_abandons, iv_value;

			// current interval
			int interval, total_servers;
			std::vector<int> no_server;
			std::vector<double> arr_cdf;

			std::mt19937_64 generator;
			std::uniform_real_distribution<double> uniform01;
			std::exponential_distribution<double> interarrivalDist;
			std::vector<std::exponential_distribution<double>> abandonDist;
			std::vector<double> numerator_tmp, ser_cdf_tmp;
	};
}

#endif
