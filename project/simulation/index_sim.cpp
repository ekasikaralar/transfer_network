#include "index_sim.h"

namespace simulation {

	// ------------------------------------------------------------------ inputs

	static std::vector<std::vector<double>> read_matrix(const std::string& filename) {
		std::vector<std::vector<double>> m;
		std::ifstream f(filename);
		if (!f.is_open()) throw std::runtime_error("Failed to open " + filename);
		std::string line;
		while (std::getline(f, line)) {
			std::vector<double> row;
			std::stringstream ss(line);
			std::string cell;
			while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
			m.push_back(row);
		}
		return m;
	}

	static std::vector<double> read_vector(const std::string& filename) {
		std::vector<double> v;
		for (const auto& row : read_matrix(filename)) v.insert(v.end(), row.begin(), row.end());
		return v;
	}

	Inputs read_inputs(const std::string& jsonFileName) {
		nlohmann::json config;
		std::ifstream file(jsonFileName);
		if (!file.is_open()) throw std::runtime_error("Unable to open config file: " + jsonFileName);
		file >> config;
		Inputs in;
		in.class_no = config["class_no"];
		in.station_no = config["station_no"];
		in.num_interval = config["num_interval"];
		in.num_iterations = config["num_iterations"];
		in.interval_len = double(config["grid_step_min"]) / 60.0;
		in.profile_iters = config.contains("profile_iters") ? int(config["profile_iters"]) : 1;
		in.policy = config["policy"].get<std::string>();
		static const std::set<std::string> valid = {"cmu", "cmu_theta", "fsf", "myopic", "myopic_theta", "value", "nn", "prices"};
		if (!valid.count(in.policy))
			throw std::runtime_error("Unknown policy '" + in.policy + "' (cmu, cmu_theta, fsf, myopic, myopic_theta, value, nn, prices)");

		in.lambda = read_vector(config["lambda_path"]);
		for (const auto& row : read_matrix(config["agents_path"])) {
			std::vector<int> r(row.size());
			for (size_t j = 0; j < row.size(); ++j) r[j] = static_cast<int>(std::llround(row[j]));
			in.no_server.push_back(r);
		}
		in.arr_cdf = read_matrix(config["arr_cdf_path"]);
		in.optimal_xi = read_matrix(config["xi_path"]);
		in.mu_hourly = read_matrix(config["mu_hourly_path"]);
		in.theta_hourly = read_vector(config["theta_hourly_path"]);
		in.cost_rate = read_vector(config["cost_rate_path"]);
		in.holding_cost = read_vector(config["holding_cost_path"]);
		in.abandon_penalty = read_vector(config["abandon_penalty_path"]);
		in.routing_P = read_matrix(config["routing_path"]);
		in.v_activity = read_matrix(config["value_path"]);
		in.vbar_activity = read_matrix(config["vbar_path"]);
		in.kappa_class = read_vector(config["kappa_path"]);

		const int I = in.class_no, K = in.station_no, M = in.num_interval;
		if ((int)in.lambda.size() != M || (int)in.no_server.size() != M || (int)in.arr_cdf.size() != M || (int)in.optimal_xi.size() != M)
			throw std::runtime_error("time-varying inputs do not have num_interval rows");
		if ((int)in.routing_P.size() != I * K) throw std::runtime_error("routing matrix must have class_no*station_no rows");
		if ((int)in.theta_hourly.size() != I || (int)in.cost_rate.size() != I || (int)in.holding_cost.size() != I
		    || (int)in.abandon_penalty.size() != I || (int)in.kappa_class.size() != I)
			throw std::runtime_error("a per class input does not have class_no entries");
		if (in.policy == "nn") in.nn = load_nn_policy(config, in.optimal_xi, in.no_server, I, K, in.interval_len);
		if (in.policy == "prices") in.nn = load_prices_policy(config, I, in.num_interval, in.interval_len);
		return in;
	}

	// ------------------------------------------------------------ replications

	int run_all(const Inputs& in, int start_iter, int end_iter, const std::string& record_file) {
		std::ofstream out(record_file);
		if (!out.is_open()) throw std::runtime_error("cannot write " + record_file);
		out << "iter,objective,value_resolved,penalty_transfers,cost_holding,cost_abandon,"
		       "value_integral,objective_vbar,arrivals,abandons,departures,transfers,"
		       "mean_queue,mean_in_system,utilization,horizon_hours,runtime_sec\n";
		const int n_iter = end_iter - start_iter;
		std::vector<std::string> lines(n_iter), maxlines(n_iter);

		#pragma omp parallel for schedule(dynamic)
		for (int iter = start_iter; iter < end_iter; ++iter) {
			auto start = std::chrono::steady_clock::now();
			Execute exec(in, iter);
			std::vector<double> res = exec.run();
			if (iter - start_iter < in.profile_iters)
				exec.save_interval_profile(record_file + ".profile." + std::to_string(iter) + ".csv");
			double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			std::string line = std::to_string(iter);
			for (double x : res) line += "," + std::to_string(x);
			line += "," + std::to_string(elapsed) + "\n";
			lines[iter - start_iter] = std::move(line);
			std::string ml = std::to_string(iter);
			for (int m : exec.class_maxima()) ml += "," + std::to_string(m);
			maxlines[iter - start_iter] = ml + "\n";
		}
		for (const auto& l : lines) out << l;
		// the largest number of callers of each class in the system during the
		// replication, used to size the state boxes of the offline networks
		std::ofstream mx(record_file + ".classmax.csv");
		mx << "iter";
		for (int i = 0; i < in.class_no; ++i) mx << ",class_" << i;
		mx << "\n";
		for (const auto& l : maxlines) mx << l;
		return 0;
	}

	// --------------------------------------------------------------- Execute

	Execute::Execute(const Inputs& in_, int seed) : in(in_), generator(seed)
	{
		const int I = in.class_no, K = in.station_no;
		uniform01 = std::uniform_real_distribution<double>(0.0, 1.0);
		abandonDist.resize(I);
		for (int i = 0; i < I; ++i) abandonDist[i] = std::exponential_distribution<double>(in.theta_hourly[i]);

		weight.assign(I, std::vector<double>(K, -1.0));     // -1 marks no activity
		for (int i = 0; i < I; ++i) {
			for (int j = 0; j < K; ++j) {
				const double mu = in.mu_hourly[i][j];
				if (mu <= 0.0) continue;
				const double c = in.cost_rate[i], th = in.theta_hourly[i], vb = in.vbar_activity[i][j];
				if      (in.policy == "cmu")          weight[i][j] = c * mu;
				else if (in.policy == "cmu_theta")    weight[i][j] = c * mu / th;
				else if (in.policy == "fsf")          weight[i][j] = mu;
				else if (in.policy == "myopic")       weight[i][j] = c + vb * mu;
				else if (in.policy == "myopic_theta") weight[i][j] = (c + vb * mu) / th;
				else if (in.policy == "value")        weight[i][j] = vb * mu;
				else if (in.policy == "nn" || in.policy == "prices") weight[i][j] = c + vb * mu;   // w_j(0), replaced at every decision
			}
		}
		numerator_tmp.assign(I, 0.0);
		ser_cdf_tmp.assign(I, 0.0);
	}

	int Execute::interval_of(double t) const {
		int iv = static_cast<int>(std::floor(t / in.interval_len));
		return std::min(std::max(iv, 0), in.num_interval - 1);
	}

	void Execute::load_interval(int iv) {
		interval = iv;
		no_server = in.no_server[iv];
		arr_cdf = in.arr_cdf[iv];
		total_servers = std::accumulate(no_server.begin(), no_server.end(), 0);
		double rate = in.lambda[iv];
		if (rate <= 0.0) rate = 1e-12;
		interarrivalDist = std::exponential_distribution<double>(rate);
	}

	int Execute::draw_destination(int cls, int station) {
		const std::vector<double>& row = in.routing_P[cls * in.station_no + station];
		double u = uniform01(generator), acc = 0.0;
		for (int l = 0; l < in.class_no; ++l) { acc += row[l]; if (u < acc) return l; }
		return -1;
	}

	double Execute::generate_service(int station) {
		double rate = 0.0;
		for (int i = 0; i < in.class_no; ++i) rate += num_in_service[station + 1][i + 1] * in.mu_hourly[i][station];
		if (rate <= 0.0) return inf;
		std::exponential_distribution<double> dist(rate);
		return dist(generator);
	}

	// ------------------------------------------------------------ queues

	void Execute::refresh_abandon_clock() {
		t_abandon = inf; class_abandon = -1; cust_abandon = -1;
		if (num_in_queue[0] == 0) return;
		int best = -1;
		for (int i = 0; i < in.class_no; ++i)
			if (next_abandon_time[i] < t_abandon) { t_abandon = next_abandon_time[i]; best = i; }
		if (best < 0) { t_abandon = inf; return; }
		class_abandon = best;
		auto it = std::min_element(abandonment_list[best].begin(), abandonment_list[best].end());
		cust_abandon = static_cast<int>(std::distance(abandonment_list[best].begin(), it));
	}

	void Execute::add_queue(double arr_time, int cls) {
		queue_list[cls].push_back(arr_time);
		double ab = arr_time + abandonDist[cls](generator);
		abandonment_list[cls].push_back(ab);
		if (ab < next_abandon_time[cls]) next_abandon_time[cls] = ab;
		refresh_abandon_clock();
	}

	// the caller at the head of the class queue enters service
	void Execute::remove_queue(int cls) {
		if (!queue_list[cls].empty()) queue_list[cls].pop_front();
		if (!abandonment_list[cls].empty()) abandonment_list[cls].erase(abandonment_list[cls].begin());
		next_abandon_time[cls] = abandonment_list[cls].empty() ? inf
			: *std::min_element(abandonment_list[cls].begin(), abandonment_list[cls].end());
		refresh_abandon_clock();
	}

	// ---------------------------------------------------------- assignment

	void Execute::start_service(int cls, int station) {
		num_in_queue[0] -= 1;
		num_in_queue[cls + 1] -= 1;
		remove_queue(cls);
		num_in_service[0][0] += 1;
		num_in_service[0][cls + 1] += 1;
		num_in_service[station + 1][0] += 1;
		num_in_service[station + 1][cls + 1] += 1;
		t_depart[station] = sim_clock + generate_service(station);
	}

	void Execute::refresh_weights() {
		if (in.policy != "nn" && in.policy != "prices") return;
		std::vector<double> p = in.nn.gradient(sim_clock, interval, num_in_system);
		in.nn.weights(p, in.mu_hourly, in.theta_hourly, in.cost_rate, in.vbar_activity, in.routing_P, weight);
	}

	// A freed agent of the pool takes the waiting class with the largest
	// weight among those the pool can serve.
	void Execute::assign_from_queue(int station) {
		refresh_weights();
		int best = -1; double best_w = -1.0;
		for (int i = 0; i < in.class_no; ++i)
			if (weight[i][station] > best_w && num_in_queue[i + 1] > 0) { best_w = weight[i][station]; best = i; }
		if (best >= 0) start_service(best, station);
	}

	void Execute::fill_idle_agents() {
		for (int j = 0; j < in.station_no; ++j) {
			while (num_in_service[j + 1][0] < no_server[j]) {
				int before = num_in_service[j + 1][0];
				assign_from_queue(j);
				if (num_in_service[j + 1][0] == before) break;
			}
		}
	}

	// --------------------------------------------------------------- events

	void Execute::handle_arrival_event(int cls) {
		num_in_system[0] += 1; num_in_system[cls + 1] += 1;
		num_arrivals[0] += 1;  num_arrivals[cls + 1] += 1;
		num_in_queue[0] += 1;  num_in_queue[cls + 1] += 1;
		add_queue(t_event, cls);
		t_arrival = sim_clock + interarrivalDist(generator);

		// the idle pool with the largest weight for this class, if any
		refresh_weights();
		int best = -1; double best_w = -1.0;
		for (int j = 0; j < in.station_no; ++j)
			if (weight[cls][j] > best_w && num_in_service[j + 1][0] < no_server[j]) { best_w = weight[cls][j]; best = j; }
		if (best >= 0) start_service(cls, best);
	}

	void Execute::handle_depart_event(int station, int cls) {
		num_in_system[0] -= 1; num_in_system[cls + 1] -= 1;
		num_departures[0] += 1; num_departures[cls + 1] += 1;
		num_in_service[0][0] -= 1; num_in_service[0][cls + 1] -= 1;
		num_in_service[station + 1][0] -= 1; num_in_service[station + 1][cls + 1] -= 1;

		int dest = draw_destination(cls, station);
		if (dest < 0) {
			value_resolved += in.v_activity[cls][station];
			iv_value[interval_of(sim_clock)] += in.v_activity[cls][station];
		} else {
			penalty_transfers += in.kappa_class[dest];
			iv_value[interval_of(sim_clock)] -= in.kappa_class[dest];
			num_transfers[cls] += 1;
			num_in_system[0] += 1; num_in_system[dest + 1] += 1;
			num_in_queue[0] += 1;  num_in_queue[dest + 1] += 1;
			add_queue(sim_clock, dest);
		}

		// the freed agent, if the pool is not above its current headcount
		if (num_in_service[station + 1][0] < no_server[station]) assign_from_queue(station);
		t_depart[station] = sim_clock + generate_service(station);
	}

	void Execute::handle_abandon_event() {
		int cls = class_abandon;
		num_in_system[0] -= 1; num_in_system[cls + 1] -= 1;
		num_abandons[0] += 1;  num_abandons[cls + 1] += 1;
		num_in_queue[0] -= 1;  num_in_queue[cls + 1] -= 1;
		cost_abandon += in.abandon_penalty[cls];
		if (cust_abandon >= 0 && cust_abandon < (int)queue_list[cls].size())
			queue_list[cls].erase(queue_list[cls].begin() + cust_abandon);
		if (cust_abandon >= 0 && cust_abandon < (int)abandonment_list[cls].size())
			abandonment_list[cls].erase(abandonment_list[cls].begin() + cust_abandon);
		next_abandon_time[cls] = abandonment_list[cls].empty() ? inf
			: *std::min_element(abandonment_list[cls].begin(), abandonment_list[cls].end());
		refresh_abandon_clock();
	}

	// --------------------------------------------------------------- output

	void Execute::save_interval_profile(const std::string& filename) {
		std::ofstream f(filename);
		f << "interval,minute,servers,mean_queue,mean_in_service,mean_idle,utilization,arrivals,abandons,value\n";
		for (int m = 0; m < in.num_interval; ++m) {
			double t = iv_time[m] > 0 ? iv_time[m] : 1e-12, q = iv_queue[m] / t, sv = iv_service[m] / t;
			f << m << "," << m * in.interval_len * 60.0 << "," << iv_servers[m] << "," << q << "," << sv << ","
			  << (iv_servers[m] - sv) << "," << (iv_servers[m] > 0 ? sv / iv_servers[m] : 0.0) << ","
			  << iv_arrivals[m] << "," << iv_abandons[m] << "," << iv_value[m] << "\n";
		}
	}

	// ------------------------------------------------------------ one run

	std::vector<double> Execute::run()
	{
		const int I = in.class_no, K = in.station_no, M = in.num_interval;
		const double total_time = M * in.interval_len;
		sim_clock = 0.0;
		load_interval(0);

		num_in_system.assign(I + 1, 0);
		num_in_queue.assign(I + 1, 0);
		num_in_service.assign(K + 1, std::vector<int>(I + 1, 0));
		queue_list.assign(I, std::deque<double>());
		abandonment_list.assign(I, std::vector<double>());
		next_abandon_time.assign(I, inf);
		num_arrivals.assign(I + 1, 0); num_abandons.assign(I + 1, 0);
		num_departures.assign(I + 1, 0); num_transfers.assign(I, 0);
		queue_integral.assign(I + 1, 0.0); system_integral.assign(I + 1, 0.0);
		max_in_system.assign(I, 0);
		value_resolved = penalty_transfers = cost_holding = cost_abandon = value_integral = 0.0;
		iv_time.assign(M, 0.0); iv_queue.assign(M, 0.0); iv_service.assign(M, 0.0); iv_servers.assign(M, 0.0);
		iv_arrivals.assign(M, 0.0); iv_abandons.assign(M, 0.0); iv_value.assign(M, 0.0);
		for (int m = 0; m < M; ++m) for (int j = 0; j < K; ++j) iv_servers[m] += in.no_server[m][j];

		// Initial state: the nominal plan of the first interval, rounded within
		// each pool by largest remainder so that every pool starts full.
		for (int j = 0; j < K; ++j) {
			std::vector<double> frac(I, 0.0);
			int assigned = 0;
			for (int i = 0; i < I; ++i) {
				double x = in.optimal_xi[0][i * K + j] * no_server[j];
				int n_ij = static_cast<int>(std::floor(x + 1e-9));
				frac[i] = x - n_ij;
				num_in_service[j + 1][i + 1] = n_ij;
				assigned += n_ij;
			}
			for (int r = assigned; r < no_server[j]; ++r) {
				int best = static_cast<int>(std::max_element(frac.begin(), frac.end()) - frac.begin());
				if (frac[best] <= 0.0) break;
				num_in_service[j + 1][best + 1] += 1;
				frac[best] = 0.0;
			}
			for (int i = 0; i < I; ++i) {
				num_in_service[j + 1][0] += num_in_service[j + 1][i + 1];
				num_in_service[0][i + 1] += num_in_service[j + 1][i + 1];
				num_in_service[0][0] += num_in_service[j + 1][i + 1];
			}
		}
		for (int i = 0; i < I; ++i) { num_in_system[i + 1] = num_in_service[0][i + 1]; num_in_system[0] += num_in_system[i + 1]; }

		t_arrival = interarrivalDist(generator);
		t_abandon = inf; class_abandon = -1; cust_abandon = -1;
		t_depart.assign(K, inf);
		for (int j = 0; j < K; ++j) t_depart[j] = sim_clock + generate_service(j);

		while (sim_clock < total_time) {
			int min_station = -1; double min_depart = inf;
			for (int j = 0; j < K; ++j) if (t_depart[j] < min_depart) { min_depart = t_depart[j]; min_station = j; }
			t_event = std::min({t_arrival, t_abandon, min_depart});
			if (t_event > total_time) t_event = total_time;
			double dt = t_event - sim_clock;

			for (int i = 0; i <= I; ++i) { queue_integral[i] += num_in_queue[i] * dt; system_integral[i] += num_in_system[i] * dt; }
			for (int i = 0; i < I; ++i) cost_holding += in.holding_cost[i] * num_in_queue[i + 1] * dt;
			for (int j = 0; j < K; ++j)
				for (int i = 0; i < I; ++i)
					if (num_in_service[j + 1][i + 1] > 0)
						value_integral += in.vbar_activity[i][j] * in.mu_hourly[i][j] * num_in_service[j + 1][i + 1] * dt;
			{
				int iv = interval_of(sim_clock);
				iv_time[iv] += dt; iv_queue[iv] += num_in_queue[0] * dt; iv_service[iv] += num_in_service[0][0] * dt;
			}

			sim_clock = t_event;
			if (sim_clock >= total_time) break;
			for (int i = 0; i < I; ++i) max_in_system[i] = std::max(max_in_system[i], num_in_system[i + 1]);

			if (t_event == t_arrival) {
				double u = uniform01(generator);
				int cls = static_cast<int>(std::lower_bound(arr_cdf.begin(), arr_cdf.end(), u) - arr_cdf.begin());
				if (cls >= I) cls = I - 1;
				iv_arrivals[interval_of(sim_clock)] += 1.0;
				handle_arrival_event(cls);
			} else if (min_station >= 0 && t_event == t_depart[min_station]) {
				double total_rate = 0.0;
				for (int i = 0; i < I; ++i) { numerator_tmp[i] = num_in_service[min_station + 1][i + 1] * in.mu_hourly[i][min_station]; total_rate += numerator_tmp[i]; }
				double s = 0.0;
				for (int i = 0; i < I; ++i) { s += numerator_tmp[i] / total_rate; ser_cdf_tmp[i] = s; }
				double u = uniform01(generator);
				int cls = static_cast<int>(std::lower_bound(ser_cdf_tmp.begin(), ser_cdf_tmp.end(), u) - ser_cdf_tmp.begin());
				if (cls >= I) cls = I - 1;
				handle_depart_event(min_station, cls);
			} else if (t_event == t_abandon) {
				iv_abandons[interval_of(sim_clock)] += 1.0;
				handle_abandon_event();
			} else {
				throw std::runtime_error("event dispatch failed");
			}

			// at an interval boundary the staffing, the class mix and the arrival
			// rate change; agents joining a pool take waiting callers, agents
			// leaving finish their current service first
			int iv_now = interval_of(sim_clock);
			if (iv_now != interval) {
				load_interval(iv_now);
				t_arrival = sim_clock + interarrivalDist(generator);
				fill_idle_agents();
			}
		}

		double arrivals = 0, abandons = 0, departures = 0, transfers = 0;
		for (int i = 0; i < I; ++i) { arrivals += num_arrivals[i + 1]; abandons += num_abandons[i + 1]; departures += num_departures[i + 1]; transfers += num_transfers[i]; }
		double servers_integral = 0.0, service_integral = 0.0;
		for (int m = 0; m < M; ++m) { servers_integral += iv_servers[m] * iv_time[m]; service_integral += iv_service[m]; }
		double objective = value_resolved - penalty_transfers - cost_holding - cost_abandon;
		double objective_vbar = value_integral - cost_holding - cost_abandon;
		return {objective, value_resolved, penalty_transfers, cost_holding, cost_abandon, value_integral, objective_vbar,
		        arrivals, abandons, departures, transfers, queue_integral[0] / total_time, system_integral[0] / total_time,
		        servers_integral > 0 ? service_integral / servers_integral : 0.0, total_time};
	}
}

int main(int argc, char** argv) {
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << " config.json start_iter end_iter output.csv\n";
		return 1;
	}
	try {
		simulation::Inputs in = simulation::read_inputs(argv[1]);
		simulation::run_all(in, std::stoi(argv[2]), std::stoi(argv[3]), argv[4]);
	} catch (std::exception& e) {
		std::cerr << "error: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Done" << std::endl;
	return 0;
}
