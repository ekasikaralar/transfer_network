# Output data for the offline approximation of the function M.
#
# For a state x and a gradient p, M is the optimal value of
#
#     minimize    sum_j w_j(p) psi_j
#     subject to  sum_{j in B(i)} psi_j >= -x_i,        i in I
#                 sum_{j in S(k)} psi_j >=  0,          k in K
#                 psi_j <= 0,                           j nonbasic
#                 psi_j <= sqrt(n) psi*_j(t),           j basic
#
# with the activity weight
#
#     w_j(p) = (mu_j - theta_b(j)) p_b(j)
#              - mu_j sum_i P_ji p_i
#              + vbar_j mu_j
#              + c_b(j).
#
# The second term is the routing feedback.  Completing activity j sends the
# caller into class i with probability P_ji, so the gradient of every
# destination class enters the weight of activity j.  The third is the value
# rate the activity earns net of rework penalties.
#
# psi_j is the shortfall of activity j against the nominal plan, so it is
# bounded above and free below: at most sqrt(n) psi*_j on a basic activity,
# where the whole nominal allocation can be withdrawn, and at most zero on a
# nonbasic one, which the nominal plan does not use and which can therefore
# only be added to.
#
# One instance is solved per (x, p) pair, and one file of pairs is read per
# grid point t_n, since the nominal allocation psi*(t_n) and the partition of
# activities into basic and nonbasic both move with t.

using JuMP, HiGHS, CSV, DataFrames, JSON, Printf, LinearAlgebra

# Paths in the config are resolved against the directory holding the config
# file, not the working directory, so the script runs identically from
# anywhere.  An absolute path in the config is left alone, and the environment
# variable DEEP_SPLITTING_DATA overrides the calibration directory when set.
config_file = abspath(length(ARGS) >= 1 ? ARGS[1] : "config.json")
config_dir  = dirname(config_file)
config = JSON.parsefile(config_file)

resolve(p) = isabspath(p) ? p : normpath(joinpath(config_dir, p))

cust_dim   = config["cust_dim"]      # I, number of caller classes
server_dim = config["server_dim"]    # K, number of server pools
scale      = config["scale"]         # n, the system scale parameter
sqrt_scale = sqrt(scale)
data_dir   = get(ENV, "DEEP_SPLITTING_DATA", resolve(config["data_dir"]))
obs_dir    = resolve(config["obs_dir"])
out_dir    = resolve(config["output_dir"])

rd(name) = Matrix(CSV.read(joinpath(data_dir, name), DataFrame, header = false))

mu       = rd(config["mu_file"])          # I x K hourly service rates
vbar     = rd(config["vbar_file"])        # I x K activity values net of rework
cost     = vec(rd(config["cost_file"]))   # I effective cost rates
theta    = vec(rd(config["theta_file"]))  # I hourly abandonment rates
agents   = rd(config["agents_file"])      # num_intervals x K pool headcounts
xi_all   = rd(config["xi_file"])          # num_intervals x (I*K) nominal plan
routing  = rd(config["routing_file"])     # (I*K) x I destination probabilities

num_intervals = size(xi_all, 1)
mkpath(out_dir)

# The activities are the (class, pool) pairs the pool is qualified to serve.
edges = vec(collect([(i, k) for i in 1:cust_dim, k in 1:server_dim if mu[i, k] > 0]))

# routing is indexed by (class, pool) flattened as (i-1)*K + k.
route_row(i, k) = routing[(i - 1) * server_dim + k, :]

"""
    activity_weights(p)

w_j(p) for every activity, as an I x K matrix with zeros off the qualification
structure.  The routing term contracts the gradient against the destination
probabilities of the activity.
"""
function activity_weights(p)
    w = zeros(cust_dim, server_dim)
    for (i, k) in edges
        w[i, k] = (mu[i, k] - theta[i]) * p[i] -
                  mu[i, k] * dot(route_row(i, k), p) +
                  vbar[i, k] * mu[i, k] +
                  cost[i]
    end
    return w
end

"""
    solve_interval(n)

Solve every instance belonging to grid point n and return the vector of optimal
values of M.  The model is built once, and only the objective coefficients and
the class right-hand sides change between instances, so the solver warm starts
from the previous basis.
"""
function solve_interval(n)
    states = Matrix(CSV.read(joinpath(obs_dir, @sprintf("state_observations_%03d.csv", n)),
                             DataFrame, header = false))
    grads  = Matrix(CSV.read(joinpath(obs_dir, @sprintf("gradient_observations_%03d.csv", n)),
                             DataFrame, header = false))
    num_obs = size(states, 1)

    xi = reshape(xi_all[n + 1, :], server_dim, cust_dim)'   # I x K for this interval
    nu = agents[n + 1, :] ./ scale
    psi_star = [xi[i, k] * nu[k] for i in 1:cust_dim, k in 1:server_dim]

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, psi[1:cust_dim, 1:server_dim])

    for (i, k) in edges
        set_upper_bound(psi[i, k],
                        xi[i, k] > 0 ? sqrt_scale * psi_star[i, k] : 0.0)
    end
    for i in 1:cust_dim, k in 1:server_dim
        if mu[i, k] == 0
            fix(psi[i, k], 0.0; force = true)
        end
    end

    con_class = @constraint(model, [i = 1:cust_dim],
                            sum(psi[i, k] for k in 1:server_dim if mu[i, k] > 0) >= 0)
    @constraint(model, [k = 1:server_dim],
                sum(psi[i, k] for i in 1:cust_dim if mu[i, k] > 0) >= 0)

    values = zeros(num_obs)
    for ins in 1:num_obs
        w = activity_weights(vec(grads[ins, :]))
        @objective(model, Min, sum(w[i, k] * psi[i, k] for (i, k) in edges))
        for i in 1:cust_dim
            set_normalized_rhs(con_class[i], -states[ins, i])
        end
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            error("interval $n instance $ins: $(termination_status(model))")
        end
        values[ins] = objective_value(model)
    end
    return values
end

for n in 0:(num_intervals - 1)
    vals = solve_interval(n)
    CSV.write(joinpath(out_dir, @sprintf("m_term_obj_value_%03d.csv", n)),
              DataFrame(obj_val = vals); header = false)
    println("interval $n done, mean M = ", sum(vals) / length(vals))
end
