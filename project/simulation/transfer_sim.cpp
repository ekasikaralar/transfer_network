#include "transfer_sim.h"

namespace simulation {

	// ------------------------------------------------------------------ inputs

	Simulation::Simulation(const std::string& jsonFileName){
		nlohmann::json config;
		std::ifstream file(jsonFileName);
		if (!file.is_open()) {
			throw std::runtime_error("Unable to open config file: " + jsonFileName);
		}
		try {
			file >> config;
		} catch (const std::exception& e) {
			throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
		}
		file.close();

		class_no = config["class_no"];
		station_no = config["station_no"];
		num_interval = config["num_interval"];
		num_iterations = config["num_iterations"];
		interval_len = double(config["grid_step_min"]) / 60.0;
		profile_iters = config.contains("profile_iters") ? int(config["profile_iters"]) : 1;
		policy = config["policy"].get<std::string>();

		// The index rules of the paper.  Each attaches a weight w_j to activity
		// j and the linear program maximizes the weighted number of callers in
		// service, which is the prelimit form of minimizing the weighted
		// shortfall from the nominal plan.
		//   cmu           w_j = c_b(j) mu_j
		//   cmu_theta     w_j = c_b(j) mu_j / theta_b(j)
		//   fsf           w_j = mu_j                         (fastest server first)
		//   myopic        w_j = c_b(j) + vbar_j mu_j
		//   myopic_theta  w_j = (c_b(j) + vbar_j mu_j) / theta_b(j)
		//   value         w_j = vbar_j mu_j                  (value rate)
		static const std::set<std::string> valid_policies =
			{"cmu", "cmu_theta", "fsf", "myopic", "myopic_theta", "value", "nn", "prices"};
		if (valid_policies.find(policy) == valid_policies.end()) {
			throw std::runtime_error("Unknown policy '" + policy +
				"' (must be one of: cmu, cmu_theta, fsf, myopic, myopic_theta, value, nn, prices)");
		}

		lambda = readVectorFromCSV(config["lambda_path"]);
		std::vector<std::vector<double>> agents = readMatrixFromCSV(config["agents_path"]);
		no_server.assign(agents.size(), std::vector<int>());
		for (size_t m = 0; m < agents.size(); ++m) {
			no_server[m].resize(agents[m].size());
			std::transform(agents[m].begin(), agents[m].end(), no_server[m].begin(),
			               [](double val) { return static_cast<int>(std::llround(val)); });
		}
		arr_cdf = readMatrixFromCSV(config["arr_cdf_path"]);
		optimal_xi = readMatrixFromCSV(config["xi_path"]);

		mu_hourly = readMatrixFromCSV(config["mu_hourly_path"]);
		theta_hourly = readVectorFromCSV(config["theta_hourly_path"]);
		cost_rate = readVectorFromCSV(config["cost_rate_path"]);
		holding_cost = readVectorFromCSV(config["holding_cost_path"]);
		abandon_penalty = readVectorFromCSV(config["abandon_penalty_path"]);

		routing_P = readMatrixFromCSV(config["routing_path"]);
		v_activity = readMatrixFromCSV(config["value_path"]);
		vbar_activity = readMatrixFromCSV(config["vbar_path"]);
		kappa_class = readVectorFromCSV(config["kappa_path"]);

		if ((int)lambda.size() != num_interval || (int)no_server.size() != num_interval
		    || (int)arr_cdf.size() != num_interval || (int)optimal_xi.size() != num_interval) {
			throw std::runtime_error("time-varying inputs do not have num_interval rows");
		}
		if ((int)routing_P.size() != class_no * station_no) {
			throw std::runtime_error("routing matrix must have class_no*station_no rows");
		}
		if ((int)theta_hourly.size() != class_no || (int)cost_rate.size() != class_no
		    || (int)holding_cost.size() != class_no || (int)abandon_penalty.size() != class_no
		    || (int)kappa_class.size() != class_no) {
			throw std::runtime_error("a per class input does not have class_no entries");
		}

		// The trained policy: gradient networks and the centering of the state.
		if (policy == "nn") nn = load_nn_policy(config, optimal_xi, no_server, class_no, station_no, interval_len);
		if (policy == "prices") nn = load_prices_policy(config, class_no, num_interval, interval_len);

		// Open a Gurobi environment once here.
		{ GRBEnv probe = GRBEnv(); }

		// an activity exists where the service rate is positive
		edges.assign(class_no, std::vector<double>(station_no, 0.0));
		for (int i = 0; i < class_no; ++i)
			for (int j = 0; j < station_no; ++j)
				edges[i][j] = (mu_hourly[i][j] > 0.0) ? 1.0 : 0.0;
	}

	Simulation::~Simulation(){
		std::cout << "Done" << std::endl;
	}

	std::vector<std::string> Simulation::splitString(const std::string& input, char delimiter) {
		std::vector<std::string> tokens;
		std::string token;
		std::istringstream tokenStream(input);
		while (std::getline(tokenStream, token, delimiter)) tokens.push_back(token);
		return tokens;
	}

	std::vector<std::vector<double>> Simulation::readMatrixFromCSV(const std::string& filename) {
		std::vector<std::vector<double>> matrix;
		std::ifstream file(filename);
		if (!file.is_open()) throw std::runtime_error("Failed to open " + filename);
		std::string line;
		while (std::getline(file, line)) {
			std::vector<double> row;
			for (const std::string& s : splitString(line, ',')) row.push_back(std::stod(s));
			matrix.push_back(row);
		}
		return matrix;
	}

	std::vector<double> Simulation::readVectorFromCSV(const std::string& filename) {
		std::vector<double> vec;
		std::ifstream file(filename);
		if (!file.is_open()) throw std::runtime_error("Failed to open " + filename);
		std::string line;
		while (std::getline(file, line)) {
			std::stringstream ss(line);
			std::string cell;
			while (std::getline(ss, cell, ',')) vec.push_back(std::stod(cell));
		}
		return vec;
	}

	// ------------------------------------------------------------ replications

	int Simulation::save(int start_iter, int end_iter, const std::string& record_file){
		int fd = open(record_file.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
		if (fd == -1) { perror("open"); return -1; }
		auto write_all = [&](const std::string& s) {
			const char* p = s.c_str();
			size_t len = s.size();
			while (len > 0) {
				ssize_t w = write(fd, p, len);
				if (w <= 0) { perror("write"); close(fd); throw std::runtime_error("write failed"); }
				p += w; len -= w;
			}
		};

		write_all("iter,objective,value_resolved,penalty_transfers,cost_holding,cost_abandon,"
		          "value_integral,objective_vbar,"
		          "arrivals,abandons,departures,transfers,mean_queue,mean_in_system,"
		          "utilization,horizon_hours,runtime_sec\n");

		// The replications are independent, each with its own linear program
		// and random number generator, so they run in parallel.  Rows are
		// buffered by index and written in order afterwards.
		const int n_iter = end_iter - start_iter;
		std::vector<std::string> lines(n_iter);

		#pragma omp parallel for schedule(dynamic)
		for (int iter = start_iter; iter < end_iter; ++iter) {
			auto start = std::chrono::steady_clock::now();
			simulation::Execute exec(
				class_no, station_no, num_interval, arr_cdf, mu_hourly, theta_hourly,
				no_server, cost_rate, holding_cost, abandon_penalty, optimal_xi, edges,
				lambda, routing_P, v_activity, vbar_activity, kappa_class,
				interval_len, policy, (policy == "nn" || policy == "prices" ? &nn : nullptr), iter);
			std::vector<double> res = exec.run();
			if (iter - start_iter < profile_iters) {
				exec.save_interval_profile(record_file + ".profile." + std::to_string(iter) + ".csv");
			}
			double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

			std::string line = std::to_string(iter);
			for (double x : res) line += "," + std::to_string(x);
			line += "," + std::to_string(elapsed) + "\n";
			lines[iter - start_iter] = std::move(line);
		}
		for (const auto& l : lines) write_all(l);
		close(fd);
		return 0;
	}

	// --------------------------------------------------------------- Execute

	Execute::Execute(int class_no_, int station_no_, int num_interval_,
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
		int seed)
		: class_no(class_no_), station_no(station_no_), num_interval(num_interval_),
		  interval_len(interval_len_),
		  mu_hourly(mu_hourly_), theta_hourly(theta_hourly_), cost_rate(cost_rate_),
		  holding_cost(holding_cost_), abandon_penalty(abandon_penalty_), edges(edges_),
		  arr_cdf_all(arr_cdf_), no_server_all(no_server_), optimal_xi_all(optimal_xi_),
		  lambda_all(lambda_), routing_P(routing_P_), v_activity(v_activity_),
		  vbar_activity(vbar_activity_), kappa_class(kappa_class_), policy(policy_),
		  nn(nn_), generator(seed)
	{
		// Gurobi settings of the original simulator: dual simplex, no presolve,
		// one thread per replication since the replications run in parallel.
		model.set(GRB_IntParam_OutputFlag, 0);
		model.set(GRB_IntParam_Method, 1);
		model.set(GRB_IntParam_Presolve, 0);
		model.set(GRB_IntParam_Threads, 1);
		model.set(GRB_DoubleParam_OptimalityTol, 1e-4);
		model.set(GRB_DoubleParam_FeasibilityTol, 1e-4);

		uniform01 = std::uniform_real_distribution<double>(0.0, 1.0);
		abandonDist.resize(class_no);
		for (int i = 0; i < class_no; ++i)
			abandonDist[i] = std::exponential_distribution<double>(theta_hourly[i]);

		// the weight of each activity under the chosen rule
		weight.assign(class_no, std::vector<double>(station_no, 0.0));
		for (int i = 0; i < class_no; ++i) {
			for (int j = 0; j < station_no; ++j) {
				if (edges[i][j] == 0.0) continue;
				const double mu = mu_hourly[i][j];
				if      (policy == "cmu")          weight[i][j] = cost_rate[i] * mu;
				else if (policy == "cmu_theta")    weight[i][j] = cost_rate[i] * mu / theta_hourly[i];
				else if (policy == "fsf")          weight[i][j] = mu;
				else if (policy == "myopic")       weight[i][j] = cost_rate[i] + vbar_activity[i][j] * mu;
				else if (policy == "myopic_theta") weight[i][j] = (cost_rate[i] + vbar_activity[i][j] * mu) / theta_hourly[i];
				else if (policy == "value")        weight[i][j] = vbar_activity[i][j] * mu;
				else if (policy == "nn" || policy == "prices") weight[i][j] = cost_rate[i] + vbar_activity[i][j] * mu;   // w_j(0), replaced at every decision
			}
		}

		num_served_buf.resize(class_no);
		optimal_queue_len_buf.resize(class_no);
		numerator_tmp.assign(class_no, 0.0);
		prob_tmp.assign(class_no, 0.0);
		ser_cdf_tmp.assign(class_no, 0.0);
		queue_init();
		load_interval(0);
	}

	Execute::~Execute() {}

	void Execute::queue_init()
	{
		queue_list.assign(class_no, std::deque<double>());
		abandonment_list.assign(class_no, std::vector<double>());
		next_abandon_time.assign(class_no, inf);
		t_abandon = inf;
		class_abandon = -1;
		cust_abandon = -1;
	}

	// ----------------------------------------------------- linear program

	// Build the assignment program once.  Variables d_ij for every activity,
	// objective sum_ij w_ij d_ij, a class row  sum_j d_ij <= x_i  and a pool row
	// sum_i d_ij <= N_j.  
	void Execute::build_model(){
		class_constraints.resize(class_no);
		station_constraints.resize(station_no);
		d.assign(class_no, std::vector<GRBVar>(station_no));
		for (int i = 0; i < class_no; ++i)
			for (int j = 0; j < station_no; ++j)
				d[i][j] = (edges[i][j] == 1.0)
				        ? model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS)
				        : model.addVar(0.0, 0.0, 0.0, GRB_CONTINUOUS);

		GRBLinExpr objective;
		for (int i = 0; i < class_no; ++i)
			for (int j = 0; j < station_no; ++j)
				if (edges[i][j] == 1.0) objective += weight[i][j] * d[i][j];
		model.setObjective(objective, GRB_MAXIMIZE);

		for (int i = 0; i < class_no; ++i) {
			GRBLinExpr lhs;
			for (int j = 0; j < station_no; ++j) if (edges[i][j] == 1.0) lhs += d[i][j];
			class_constraints[i] = model.addConstr(lhs <= num_in_system[i + 1]);
		}
		for (int j = 0; j < station_no; ++j) {
			GRBLinExpr lhs;
			for (int i = 0; i < class_no; ++i) if (edges[i][j] == 1.0) lhs += d[i][j];
			station_constraints[j] = model.addConstr(lhs <= no_server[j]);
		}
		model.update();
		optimal_d.assign(class_no, std::vector<double>(station_no, 0.0));
	}

	// Preemptive resume: every caller in the system, waiting or in service,
	// can be reassigned, so the class rows carry the number in the system.
	// nn: the weights depend on the state through the gradient of the value
	// function, so they are recomputed before every decision
	void Execute::refresh_weights() {
		if (nn == nullptr) return;
		std::vector<double> p = nn->gradient(sim_clock, interval, num_in_system);
		nn->weights(p, mu_hourly, theta_hourly, cost_rate, vbar_activity, routing_P, weight);
		for (int i = 0; i < class_no; ++i)
			for (int j = 0; j < station_no; ++j)
				if (edges[i][j] == 1.0) d[i][j].set(GRB_DoubleAttr_Obj, weight[i][j]);
	}

	const std::vector<std::vector<double>>& Execute::linear_program() {
		refresh_weights();
		for (int i = 0; i < class_no; ++i)
			class_constraints[i].set(GRB_DoubleAttr_RHS, num_in_system[i + 1]);
		model.optimize();
		for (int i = 0; i < class_no; ++i)
			for (int j = 0; j < station_no; ++j)
				optimal_d[i][j] = (edges[i][j] == 1.0) ? d[i][j].get(GRB_DoubleAttr_X) : 0.0;
		return optimal_d;
	}

	void Execute::refresh_station_capacity() {
		for (int j = 0; j < station_no; ++j)
			station_constraints[j].set(GRB_DoubleAttr_RHS, no_server[j]);
	}

	// ---------------------------------------------- time varying primitives

	int Execute::interval_of(double t) const {
		int iv = static_cast<int>(std::floor(t / interval_len));
		if (iv < 0) iv = 0;
		if (iv >= num_interval) iv = num_interval - 1;
		return iv;
	}

	void Execute::load_interval(int iv) {
		interval = iv;
		no_server = no_server_all[iv];
		arr_cdf = arr_cdf_all[iv];
		total_servers = 0;
		for (int j = 0; j < station_no; ++j) total_servers += no_server[j];
		double rate = lambda_all[iv];
		if (rate <= 0.0) rate = 1e-12;
		interarrivalDist = std::exponential_distribution<double>(rate);
	}

	// ------------------------------------------------------------ transfers

	int Execute::draw_destination(int cls, int station) {
		const std::vector<double>& row = routing_P[cls * station_no + station];
		double u = uniform01(generator);
		double acc = 0.0;
		for (int l = 0; l < class_no; ++l) {
			acc += row[l];
			if (u < acc) return l;
		}
		return -1;
	}

	void Execute::inject_transfer(int dest_cls) {
		num_in_system[0] += 1;
		num_in_system[dest_cls + 1] += 1;
		num_in_queue[0] += 1;
		num_in_queue[dest_cls + 1] += 1;
		add_queue(sim_clock, dest_cls);
	}

	// -------------------------------------------------------- random draws

	double Execute::generate_interarrival(){ return interarrivalDist(generator); }

	double Execute::generate_abandon(int cls) { return abandonDist[cls](generator); }

	// The pool's departure clock runs at the total service rate of the callers
	// in service there.
	double Execute::generate_service(int station){
		double service_rate = 0.0;
		for (int i = 0; i < class_no; ++i)
			service_rate += num_in_service[station + 1][i + 1] * mu_hourly[i][station];
		if (service_rate <= 0.0) return inf;
		std::exponential_distribution<double> dist(service_rate);
		return dist(generator);
	}

	// ------------------------------------------------------- assignment

	// Solve the program and move callers between queue and service so that
	// the numbers in service match the solution.
	void Execute::system_arrangement()
	{
		const std::vector<std::vector<double>>& opt = linear_program();

		for (int i = 0; i < class_no; ++i) {
			int served_i = 0;
			for (int j = 0; j < station_no; ++j)
				served_i += static_cast<int>(std::lround(opt[i][j]));
			num_served_buf[i] = served_i;
			optimal_queue_len_buf[i] = num_in_system[i + 1] - served_i;
			int diff = num_in_queue[i + 1] - optimal_queue_len_buf[i];
			num_in_queue[0] -= diff;
			num_in_queue[i + 1] -= diff;
			if (diff > 0) {
				for (int k = 0; k < diff; ++k) remove_queue(i);
			} else if (diff < 0) {
				for (int k = 0; k < -diff; ++k) add_queue(t_event, i);
			}
		}

		num_in_service[0][0] = 0;
		for (int j = 0; j < station_no; ++j) num_in_service[j + 1][0] = 0;
		for (int i = 0; i < class_no; ++i) num_in_service[0][i + 1] = 0;
		for (int i = 0; i < class_no; ++i) {
			for (int j = 0; j < station_no; ++j) {
				int x_ij = static_cast<int>(std::lround(opt[i][j]));
				num_in_service[j + 1][i + 1] = x_ij;
				num_in_service[j + 1][0] += x_ij;
				num_in_service[0][i + 1] += x_ij;
				num_in_service[0][0] += x_ij;
			}
		}
	}

	// Re-solve when there is both idle capacity and waiting work, then
	// reschedule the departure clocks of the pools.
	void Execute::rearrange_system() {
		bool has_idle_server = (num_in_service[0][0] < total_servers);
		bool has_waiting_jobs = (num_in_queue[0] > 0);
		if (has_idle_server && has_waiting_jobs) system_arrangement();

		if (num_in_service[0][0] > 0) {
			for (int j = 0; j < station_no; ++j) t_depart[j] = sim_clock + generate_service(j);
		} else {
			for (int j = 0; j < station_no; ++j) t_depart[j] = inf;
		}
	}

	// A lone arrival that finds an idle agent goes to the idle pool with the
	// largest weight of the rule, which is what the program would choose.
	int Execute::choose_station_for_arrival(int cls) {
		int best_station = -1;
		double best_w = -1.0;
		for (int j = 0; j < station_no; ++j) {
			if (edges[cls][j] == 0.0) continue;
			if (num_in_service[j + 1][0] >= no_server[j]) continue;
			if (weight[cls][j] > best_w) { best_w = weight[cls][j]; best_station = j; }
		}
		return best_station;
	}

	// --------------------------------------------------------------- events

	void Execute::handle_arrival_event(int cls){
		num_in_system[0] += 1;
		num_in_system[cls + 1] += 1;
		num_arrivals[0] += 1;
		num_arrivals[cls + 1] += 1;
		num_in_queue[0] += 1;
		num_in_queue[cls + 1] += 1;
		add_queue(t_event, cls);
		t_arrival = sim_clock + generate_interarrival();

		bool has_idle_server = (num_in_service[0][0] < total_servers);
		bool is_single_in_q = (num_in_queue[0] == 1);
		if (has_idle_server && is_single_in_q) {
			if (nn != nullptr) {
				std::vector<double> p = nn->gradient(sim_clock, interval, num_in_system);
				nn->weights(p, mu_hourly, theta_hourly, cost_rate, vbar_activity, routing_P, weight);
			}
			int best_station = choose_station_for_arrival(cls);
			if (best_station != -1) {
				num_in_queue[0] -= 1;
				num_in_queue[cls + 1] -= 1;
				remove_queue(cls);
				num_in_service[0][0] += 1;
				num_in_service[0][cls + 1] += 1;
				num_in_service[best_station + 1][0] += 1;
				num_in_service[best_station + 1][cls + 1] += 1;
				t_depart[best_station] = sim_clock + generate_service(best_station);
				return;
			}
		}
		rearrange_system();
	}

	// A completion of activity (cls, station) either resolves the request,
	// generating v_j, or routes the caller to a rework class l, costing
	// kappa_l and putting the caller at the back of that class's queue.
	void Execute::handle_depart_event(int station, int cls){
		num_in_system[0] -= 1;
		num_in_system[cls + 1] -= 1;
		num_departures[0] += 1;
		num_departures[cls + 1] += 1;
		num_in_service[0][0] -= 1;
		num_in_service[0][cls + 1] -= 1;
		num_in_service[station + 1][0] -= 1;
		num_in_service[station + 1][cls + 1] -= 1;

		int dest = draw_destination(cls, station);
		if (dest < 0) {
			value_resolved += v_activity[cls][station];
			iv_value[interval_of(sim_clock)] += v_activity[cls][station];
			num_resolved[cls] += 1;
		} else {
			penalty_transfers += kappa_class[dest];
			iv_value[interval_of(sim_clock)] -= kappa_class[dest];
			num_transfers[cls] += 1;
			inject_transfer(dest);
		}
		rearrange_system();
	}

	void Execute::handle_abandon_event() {
		num_in_system[0] -= 1;
		num_in_system[class_abandon + 1] -= 1;
		num_abandons[0] += 1;
		num_abandons[class_abandon + 1] += 1;
		num_in_queue[0] -= 1;
		num_in_queue[class_abandon + 1] -= 1;
		cost_abandon += abandon_penalty[class_abandon];

		if (cust_abandon >= 0 && cust_abandon < (int)queue_list[class_abandon].size())
			queue_list[class_abandon].erase(queue_list[class_abandon].begin() + cust_abandon);
		if (cust_abandon >= 0 && cust_abandon < (int)abandonment_list[class_abandon].size())
			abandonment_list[class_abandon].erase(abandonment_list[class_abandon].begin() + cust_abandon);

		next_abandon_time[class_abandon] = abandonment_list[class_abandon].empty() ? inf
			: *std::min_element(abandonment_list[class_abandon].begin(), abandonment_list[class_abandon].end());

		t_abandon = inf; class_abandon = -1; cust_abandon = -1;
		if (num_in_queue[0] > 0) {
			double best_time = inf; int best_class = -1;
			for (int i = 0; i < class_no; ++i)
				if (next_abandon_time[i] < best_time) { best_time = next_abandon_time[i]; best_class = i; }
			if (best_class != -1) {
				t_abandon = best_time;
				class_abandon = best_class;
				auto it = std::min_element(abandonment_list[best_class].begin(), abandonment_list[best_class].end());
				cust_abandon = static_cast<int>(std::distance(abandonment_list[best_class].begin(), it));
			}
		}
	}

	void Execute::add_queue(double arr_time, int cls) {
		queue_list[cls].push_back(arr_time);
		double abandon_time = arr_time + generate_abandon(cls);
		abandonment_list[cls].push_back(abandon_time);
		if (abandon_time < next_abandon_time[cls]) next_abandon_time[cls] = abandon_time;

		double best_time = inf; int best_class = -1;
		for (int i = 0; i < class_no; ++i)
			if (next_abandon_time[i] < best_time) { best_time = next_abandon_time[i]; best_class = i; }
		if (best_class == -1) { t_abandon = inf; class_abandon = -1; cust_abandon = -1; return; }
		t_abandon = best_time;
		class_abandon = best_class;
		auto it = std::min_element(abandonment_list[best_class].begin(), abandonment_list[best_class].end());
		cust_abandon = static_cast<int>(std::distance(abandonment_list[best_class].begin(), it));
	}

	void Execute::remove_queue(int cls) {
		if (!queue_list[cls].empty()) queue_list[cls].pop_front();
		if (!abandonment_list[cls].empty()) abandonment_list[cls].erase(abandonment_list[cls].begin());
		next_abandon_time[cls] = abandonment_list[cls].empty() ? inf
			: *std::min_element(abandonment_list[cls].begin(), abandonment_list[cls].end());

		if (num_in_queue[0] == 0) { t_abandon = inf; class_abandon = -1; cust_abandon = -1; return; }
		double best_time = inf; int best_class = -1;
		for (int i = 0; i < class_no; ++i)
			if (next_abandon_time[i] < best_time) { best_time = next_abandon_time[i]; best_class = i; }
		if (best_class == -1 || abandonment_list[best_class].empty()) {
			t_abandon = inf; class_abandon = -1; cust_abandon = -1; return;
		}
		t_abandon = best_time;
		class_abandon = best_class;
		auto it = std::min_element(abandonment_list[best_class].begin(), abandonment_list[best_class].end());
		cust_abandon = static_cast<int>(std::distance(abandonment_list[best_class].begin(), it));
	}

	// --------------------------------------------------------------- output

	void Execute::save_interval_profile(const std::string& filename){
		std::ofstream f(filename);
		f << "interval,minute,servers,mean_queue,mean_in_service,mean_idle,utilization,arrivals,abandons,value\n";
		for (int m = 0; m < num_interval; ++m) {
			double t = iv_time[m] > 0 ? iv_time[m] : 1e-12;
			double q = iv_queue[m] / t;
			double sv = iv_service[m] / t;
			f << m << "," << m * interval_len * 60.0 << "," << iv_servers[m] << ","
			  << q << "," << sv << "," << (iv_servers[m] - sv) << ","
			  << (iv_servers[m] > 0 ? sv / iv_servers[m] : 0.0) << ","
			  << iv_arrivals[m] << "," << iv_abandons[m] << "," << iv_value[m] << "\n";
		}
	}

	// ------------------------------------------------------------ one run

	std::vector<double> Execute::run()
	{
		const double total_time = num_interval * interval_len;
		sim_clock = 0.0;
		load_interval(0);

		num_in_system.assign(class_no + 1, 0);
		num_in_service.assign(station_no + 1, std::vector<int>(class_no + 1, 0));
		num_in_queue.assign(class_no + 1, 0);
		num_arrivals.assign(class_no + 1, 0);
		num_abandons.assign(class_no + 1, 0);
		num_departures.assign(class_no + 1, 0);
		num_transfers.assign(class_no, 0);
		num_resolved.assign(class_no, 0);
		queue_integral.assign(class_no + 1, 0.0);
		system_integral.assign(class_no + 1, 0.0);
		value_resolved = 0.0;
		penalty_transfers = 0.0;
		cost_holding = 0.0;
		cost_abandon = 0.0;
		value_integral = 0.0;

		iv_time.assign(num_interval, 0.0);
		iv_queue.assign(num_interval, 0.0);
		iv_service.assign(num_interval, 0.0);
		iv_servers.assign(num_interval, 0.0);
		iv_arrivals.assign(num_interval, 0.0);
		iv_abandons.assign(num_interval, 0.0);
		iv_value.assign(num_interval, 0.0);
		for (int m = 0; m < num_interval; ++m)
			for (int j = 0; j < station_no; ++j) iv_servers[m] += no_server_all[m][j];

		t_arrival = generate_interarrival();
		t_depart.assign(station_no, inf);
		t_abandon = inf;

		// Initial state: the nominal plan of the first interval, which fills
		// every pool, with an empty queue.  Every pool starts exactly full.
		for (int j = 0; j < station_no; ++j) {
			std::vector<double> frac(class_no, 0.0);
			int assigned = 0;
			for (int i = 0; i < class_no; ++i) {
				double x = optimal_xi_all[0][i * station_no + j] * no_server[j];
				int n_ij = static_cast<int>(std::floor(x + 1e-9));
				frac[i] = x - n_ij;
				num_in_service[j + 1][i + 1] = n_ij;
				assigned += n_ij;
			}
			for (int r = assigned; r < no_server[j]; ++r) {
				int best = static_cast<int>(std::max_element(frac.begin(), frac.end()) - frac.begin());
				if (frac[best] <= 0.0) break;        // nothing left to allocate to
				num_in_service[j + 1][best + 1] += 1;
				frac[best] = 0.0;
			}
			for (int i = 0; i < class_no; ++i) {
				num_in_service[j + 1][0] += num_in_service[j + 1][i + 1];
				num_in_service[0][i + 1] += num_in_service[j + 1][i + 1];
				num_in_service[0][0] += num_in_service[j + 1][i + 1];
			}
		}
		for (int i = 0; i < class_no; ++i) {
			num_in_system[i + 1] = num_in_service[0][i + 1];
			num_in_system[0] += num_in_system[i + 1];
		}

		build_model();
		for (int j = 0; j < station_no; ++j) t_depart[j] = sim_clock + generate_service(j);

		while (sim_clock < total_time) {
			int min_depart_station = -1;
			double min_depart_time = inf;
			for (int j = 0; j < station_no; ++j)
				if (t_depart[j] < min_depart_time) { min_depart_time = t_depart[j]; min_depart_station = j; }
			t_event = std::min({t_arrival, t_abandon, min_depart_time});
			if (t_event > total_time) t_event = total_time;
			double dt = t_event - sim_clock;

			// time integrals of the queues, the holding cost, and the profile
			for (int i = 0; i <= class_no; ++i) {
				queue_integral[i] += num_in_queue[i] * dt;
				system_integral[i] += num_in_system[i] * dt;
			}
			for (int i = 0; i < class_no; ++i)
				cost_holding += holding_cost[i] * num_in_queue[i + 1] * dt;
			// the value term of the paper's objective, sum_j vbar_j mu_j psi_j,
			// with psi_j the number of agents on activity j
			for (int k = 0; k < station_no; ++k)
				for (int i = 0; i < class_no; ++i)
					if (num_in_service[k + 1][i + 1] > 0)
						value_integral += vbar_activity[i][k] * mu_hourly[i][k] * num_in_service[k + 1][i + 1] * dt;
			{
				int iv = interval_of(sim_clock);
				iv_time[iv] += dt;
				iv_queue[iv] += num_in_queue[0] * dt;
				iv_service[iv] += num_in_service[0][0] * dt;
			}

			sim_clock = t_event;
			if (sim_clock >= total_time) break;

			if (t_event == t_arrival) {
				double u = uniform01(generator);
				auto low = std::lower_bound(arr_cdf.begin(), arr_cdf.end(), u);
				int cls = static_cast<int>(low - arr_cdf.begin());
				if (cls >= class_no) cls = class_no - 1;
				iv_arrivals[interval_of(sim_clock)] += 1.0;
				handle_arrival_event(cls);
			} else if (min_depart_station != -1 && t_event == t_depart[min_depart_station]) {
				// which class completed at this pool, in proportion to the rates
				int station = min_depart_station;
				double total_rate = 0.0;
				for (int i = 0; i < class_no; ++i) {
					numerator_tmp[i] = num_in_service[station + 1][i + 1] * mu_hourly[i][station];
					total_rate += numerator_tmp[i];
				}
				double s = 0.0;
				for (int i = 0; i < class_no; ++i) { s += numerator_tmp[i] / total_rate; ser_cdf_tmp[i] = s; }
				double u = uniform01(generator);
				auto low = std::lower_bound(ser_cdf_tmp.begin(), ser_cdf_tmp.end(), u);
				int cls = static_cast<int>(low - ser_cdf_tmp.begin());
				if (cls >= class_no) cls = class_no - 1;
				handle_depart_event(station, cls);
			} else if (t_event == t_abandon) {
				iv_abandons[interval_of(sim_clock)] += 1.0;
				handle_abandon_event();
			} else {
				throw std::runtime_error("event dispatch failed");
			}

			// at an interval boundary the staffing, the class mix and the
			// arrival rate change, so the pool rows and the assignment are refreshed
			int iv_now = interval_of(sim_clock);
			if (iv_now != interval) {
				load_interval(iv_now);
				refresh_station_capacity();
				t_arrival = sim_clock + generate_interarrival();
				rearrange_system();
			}
		}

		double arrivals = 0, abandons = 0, departures = 0, transfers = 0;
		for (int i = 0; i < class_no; ++i) {
			arrivals += num_arrivals[i + 1];
			abandons += num_abandons[i + 1];
			departures += num_departures[i + 1];
			transfers += num_transfers[i];
		}
		double servers_integral = 0.0;
		for (int m = 0; m < num_interval; ++m) servers_integral += iv_servers[m] * iv_time[m];
		double service_integral = 0.0;
		for (int m = 0; m < num_interval; ++m) service_integral += iv_service[m];

		// realized objective, and the objective with the value term replaced by
		// its integral form, whose mean is the same but whose noise is smaller
		double objective = value_resolved - penalty_transfers - cost_holding - cost_abandon;
		double objective_vbar = value_integral - cost_holding - cost_abandon;
		return {objective, value_resolved, penalty_transfers, cost_holding, cost_abandon,
		        value_integral, objective_vbar,
		        arrivals, abandons, departures, transfers,
		        queue_integral[0] / total_time, system_integral[0] / total_time,
		        servers_integral > 0 ? service_integral / servers_integral : 0.0,
		        total_time};
	}
}

int main(int argc, char** argv){
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << " config.json start_iter end_iter output.csv\n";
		return 1;
	}
	std::string jsonFileName = argv[1];
	int start_iter = std::stoi(argv[2]);
	int end_iter = std::stoi(argv[3]);   // end is exclusive
	std::string record_file = argv[4];
	try {
		simulation::Simulation simObj(jsonFileName);
		simObj.save(start_iter, end_iter, record_file);
	} catch (GRBException& e) {
		std::cerr << "Gurobi error " << e.getErrorCode() << ": " << e.getMessage() << std::endl;
		return 2;
	} catch (std::exception& e) {
		std::cerr << "error: " << e.what() << std::endl;
		return 1;
	}
	return 0;
}
