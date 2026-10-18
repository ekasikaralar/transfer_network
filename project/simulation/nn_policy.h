#ifndef NN_POLICY_H
#define NN_POLICY_H

// The trained policy of the deep splitting solver.
//
// At time t and state X the policy evaluates the gradient network of the
// current grid point at the centered state
//     xhat_i = (X_i - x_nom_i(t)) / sqrt(n),      x_nom_i(t) = sum_j xi_ij(t) N_j(t),
// obtaining p = dV/dx, and attaches to activity (i, j) the weight
//     w_ij(p) = (mu_ij - theta_i) p_i - mu_ij sum_l P_(ij),l p_l + vbar_ij mu_ij + c_i,
// the objective of the linear program defining M.  With p = 0 this is the
// myopic rule, so a network with a zero output layer reproduces it exactly.
//
// Networks: Z_Net of the solver, num_hidden_layers linear layers of width
// units with the activation after each, then an output layer of width I with
// no bias.  The solver writes them as text, z_network<n>_w0..w<L>.txt and
// z_network<n>_b0..b<L-1>.txt, PyTorch layout, out by in.

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace simulation
{
	class GradientNetwork
	{
		public:
			void load(const std::string& dir, const std::string& stem, int hidden_layers,
			          const std::string& activation, double slope) {
				act = activation; leaky = slope;
				W.clear(); b.clear();
				for (int l = 0; l < hidden_layers; ++l) {
					W.push_back(read_matrix(dir + stem + "_w" + std::to_string(l) + ".txt"));
					b.push_back(read_vector(dir + stem + "_b" + std::to_string(l) + ".txt"));
				}
				W.push_back(read_matrix(dir + stem + "_w" + std::to_string(hidden_layers) + ".txt"));
			}

			// y = W_L a_L, a_{l+1} = act(W_l a_l + b_l), a_0 = x
			std::vector<double> forward(const std::vector<double>& x) const {
				std::vector<double> a = x;
				for (size_t l = 0; l + 1 < W.size(); ++l) {
					std::vector<double> z(W[l].size());
					for (size_t r = 0; r < W[l].size(); ++r) {
						double s = b[l][r];
						const std::vector<double>& row = W[l][r];
						for (size_t c = 0; c < row.size(); ++c) s += row[c] * a[c];
						z[r] = activate(s);
					}
					a.swap(z);
				}
				const auto& Wo = W.back();
				std::vector<double> y(Wo.size());
				for (size_t r = 0; r < Wo.size(); ++r) {
					double s = 0.0;
					for (size_t c = 0; c < Wo[r].size(); ++c) s += Wo[r][c] * a[c];
					y[r] = s;
				}
				return y;
			}

		private:
			std::vector<std::vector<std::vector<double>>> W;
			std::vector<std::vector<double>> b;
			std::string act;
			double leaky = 0.0;

			double activate(double v) const {
				if (act == "eLU" || act == "ELU") return v > 0 ? v : std::expm1(v);
				if (act == "ReLU") return v > 0 ? v : 0.0;
				if (act == "LeakyReLU") return v > 0 ? v : leaky * v;
				if (act == "SELU") { const double a = 1.6732632423543772, s = 1.0507009873554805; return s * (v > 0 ? v : a * std::expm1(v)); }
				if (act == "GELU") return 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
				throw std::runtime_error("unknown activation '" + act + "'");
			}
			static std::vector<std::vector<double>> read_matrix(const std::string& f) {
				std::ifstream in(f);
				if (!in.is_open()) throw std::runtime_error("cannot open " + f);
				std::vector<std::vector<double>> m;
				std::string line;
				while (std::getline(in, line)) {
					if (line.empty()) continue;
					std::vector<double> row;
					std::stringstream ss(line);
					std::string cell;
					while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
					m.push_back(row);
				}
				return m;
			}
			static std::vector<double> read_vector(const std::string& f) {
				std::vector<double> v;
				for (const auto& row : read_matrix(f)) v.insert(v.end(), row.begin(), row.end());
				return v;
			}
	};

	// Everything the policy needs besides the state.  Built once per run and
	// shared read only by the replications.
	struct NNPolicy
	{
		std::vector<GradientNetwork> nets;            // one per splitting interval
		std::vector<std::vector<double>> prices;       // policy prices: model interval x class, used when nets is empty
		std::vector<std::vector<double>> x_nom;        // model interval x class, callers
		double sqrt_n = 10.0;
		double total_time = 8.0;                       // hours

		// index of the network in force at time t
		int network_index(double t) const {
			int idx = static_cast<int>(std::floor(t / total_time * nets.size()));
			if (idx < 0) idx = 0;
			if (idx >= (int)nets.size()) idx = (int)nets.size() - 1;
			return idx;
		}

		// p = dV/dx at the centered state.
		// Under policy prices there is no network and p is the table row of the
		// model interval containing t, a price per class that does not depend on
		// the state. 
		std::vector<double> gradient(double t, int interval, const std::vector<int>& num_in_system) const {
			if (nets.empty()) {
				int m = static_cast<int>(std::floor(t / total_time * prices.size()));
				if (m < 0) m = 0;
				if (m >= (int)prices.size()) m = (int)prices.size() - 1;
				return prices[m];
			}
			const int I = (int)x_nom[interval].size();
			std::vector<double> xhat(I);
			for (int i = 0; i < I; ++i) xhat[i] = (num_in_system[i + 1] - x_nom[interval][i]) / sqrt_n;
			std::vector<double> p = nets[network_index(t)].forward(xhat);
			return p;
		}

		// the activity weights w_ij(p), -1 where there is no activity
		void weights(const std::vector<double>& p,
		             const std::vector<std::vector<double>>& mu, const std::vector<double>& theta,
		             const std::vector<double>& cost, const std::vector<std::vector<double>>& vbar,
		             const std::vector<std::vector<double>>& routing_P,
		             std::vector<std::vector<double>>& w) const {
			const int I = (int)mu.size(), K = (int)mu[0].size();
			for (int i = 0; i < I; ++i) {
				for (int j = 0; j < K; ++j) {
					if (mu[i][j] <= 0.0) { w[i][j] = -1.0; continue; }
					const std::vector<double>& P = routing_P[i * K + j];
					double transfer = 0.0;
					for (int l = 0; l < I; ++l) transfer += P[l] * p[l];
					w[i][j] = (mu[i][j] - theta[i]) * p[i] - mu[i][j] * transfer + vbar[i][j] * mu[i][j] + cost[i];
				}
			}
		}
	};

	// Policy prices: the weight formula w_ij(p) with p read from a table, one
	// row per model interval and one column per class, instead of a network.
	// The fluid price rule is this with p = minus the planning problem's duals.
	template <class Json>
	NNPolicy load_prices_policy(const Json& config, int class_no, int num_interval, double interval_len) {
		NNPolicy nn;
		nn.total_time = num_interval * interval_len;
		std::string path = config["prices_path"].template get<std::string>();
		std::ifstream in(path);
		if (!in.is_open()) throw std::runtime_error("cannot open " + path);
		std::string line;
		while (std::getline(in, line)) {
			if (line.empty()) continue;
			std::vector<double> row;
			std::stringstream ss(line);
			std::string cell;
			while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
			if ((int)row.size() != class_no) throw std::runtime_error("prices_path: a row has " + std::to_string(row.size()) + " columns, expected " + std::to_string(class_no));
			nn.prices.push_back(row);
		}
		if ((int)nn.prices.size() != num_interval)
			throw std::runtime_error("prices_path has " + std::to_string(nn.prices.size()) + " rows, expected one per model interval, " + std::to_string(num_interval));
		return nn;
	}

	// Reads the nn_* keys of a config and the networks.  xi and agents are the
	// model grid inputs, used for the centering.
	template <class Json>
	NNPolicy load_nn_policy(const Json& config, const std::vector<std::vector<double>>& xi,
	                        const std::vector<std::vector<int>>& agents, int class_no, int station_no,
	                        double interval_len) {
		NNPolicy nn;
		std::string dir = config["nn_dir"].template get<std::string>();
		if (!dir.empty() && dir.back() != '/') dir += '/';
		int n_nets = config.contains("nn_interval") ? int(config["nn_interval"]) : (int)xi.size();
		int hidden = config.contains("num_hidden_layers") ? int(config["num_hidden_layers"]) : 4;
		std::string activation = config.contains("activation") ? config["activation"].template get<std::string>() : "eLU";
		double slope = config.contains("slope") ? double(config["slope"]) : 0.0;
		double n_scale = config.contains("n_scale") ? double(config["n_scale"]) : 100.0;
		nn.sqrt_n = std::sqrt(n_scale);
		nn.total_time = xi.size() * interval_len;
		// nn_interval must equal the number of networks the solver wrote, since
		// network_index spreads the networks evenly over the horizon
		{
			std::ifstream probe(dir + "z_network" + std::to_string(n_nets) + "_w0.txt");
			if (probe.is_open())
				throw std::runtime_error("nn_interval is " + std::to_string(n_nets) + " but " + dir +
				                         " holds more networks; set nn_interval to the number of networks");
		}
		nn.nets.resize(n_nets);
		for (int n = 0; n < n_nets; ++n) nn.nets[n].load(dir, "z_network" + std::to_string(n), hidden, activation, slope);
		nn.x_nom.assign(xi.size(), std::vector<double>(class_no, 0.0));
		for (size_t m = 0; m < xi.size(); ++m)
			for (int i = 0; i < class_no; ++i)
				for (int j = 0; j < station_no; ++j)
					nn.x_nom[m][i] += xi[m][i * station_no + j] * agents[m][j];
		return nn;
	}
}

#endif
