# Output data for the offline approximation of the function D.
#
# D is the drift the reference policy contributes to the state,
#
#     D_i(t, x) = sum_{j in B(i)} (mu_j - theta_b(j)) psi_j
#                 - sum_{j in J} P_ji mu_j psi_j,
#
# where psi solves the reference linear program
#
#     minimize    sum_j wtilde_j psi_j
#     subject to  sum_{j in B(i)} psi_j >= -x_i,        i in I
#                 sum_{j in S(k)} psi_j >=  0,          k in K
#                 psi_j <= 0,                           j nonbasic
#                 psi_j <= sqrt(n) psi*_j(t),           j basic
#
# over the same feasible set as the M term.  psi_j is a shortfall, so
# minimizing the weighted shortfall is the way an index rule expresses "serve
# the activities with the largest weight first".
#
# The index rules and their weights:
#
#     cmu           wtilde_j = c_b(j) mu_j
#     cmu_theta     wtilde_j = c_b(j) mu_j / theta_b(j)
#     fsf           wtilde_j = mu_j
#     myopic        wtilde_j = c_b(j) + vbar_j mu_j
#     myopic_theta  wtilde_j = (c_b(j) + vbar_j mu_j) / theta_b(j)
#     value         wtilde_j = vbar_j mu_j
#
# The weights do not depend on the state or on t, so the objective is set once
# and only the class right-hand sides change between instances.
#
# The second sum in D is the routing feedback.  A completion of activity j
# sends the caller into class i with probability P_ji, so every activity
# contributes to the drift of every class it can route into, not only to the
# class it serves.  

using JuMP, HiGHS, CSV, DataFrames, JSON, Printf, LinearAlgebra

config_file = abspath(length(ARGS) >= 1 ? ARGS[1] : "config.json")
config_dir  = dirname(config_file)
config = JSON.parsefile(config_file)

resolve(p) = isabspath(p) ? p : normpath(joinpath(config_dir, p))

cust_dim   = config["cust_dim"]      # I, number of caller classes
server_dim = config["server_dim"]    # K, number of server pools
scale      = config["scale"]         # n, the system scale parameter
sqrt_scale = sqrt(scale)
policy     = config["policy"]
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
    reference_weights(policy)

wtilde_j for the chosen index rule.
"""
function reference_weights(policy)
    w = zeros(cust_dim, server_dim)
    for (i, k) in edges
        w[i, k] =
            policy == "cmu"          ? cost[i] * mu[i, k] :
            policy == "cmu_theta"    ? cost[i] * mu[i, k] / theta[i] :
            policy == "fsf"          ? mu[i, k] :
            policy == "myopic"       ? cost[i] + vbar[i, k] * mu[i, k] :
            policy == "myopic_theta" ? (cost[i] + vbar[i, k] * mu[i, k]) / theta[i] :
            policy == "value"        ? vbar[i, k] * mu[i, k] :
            error("unsupported policy $policy")
    end
    return w
end

"""
    drift(psi)

D_i for every class, given the reference allocation.  The first sum is the
class's own service and abandonment, the second the callers routed into it by
completions anywhere in the network.
"""
function drift(psi)
    d = zeros(cust_dim)
    for (i, k) in edges
        d[i] += (mu[i, k] - theta[i]) * psi[i, k]
        d .-= route_row(i, k) .* (mu[i, k] * psi[i, k])
    end
    return d
end

"""
    solve_interval(n, w)

Solve every instance belonging to grid point n and return a num_obs x I matrix
of drift vectors.  The model is built once and only the class right-hand sides
change between instances, so the solver warm starts from the previous basis.
"""
function solve_interval(n, w)
    states = Matrix(CSV.read(joinpath(obs_dir, @sprintf("state_observations_%03d.csv", n)),
                             DataFrame, header = false))
    num_obs = size(states, 1)

    xi = reshape(xi_all[n + 1, :], server_dim, cust_dim)'   # I x K for this interval
    nu = agents[n + 1, :] ./ scale

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, psi[1:cust_dim, 1:server_dim])

    for (i, k) in edges
        set_upper_bound(psi[i, k],
                        xi[i, k] > 0 ? sqrt_scale * xi[i, k] * nu[k] : 0.0)
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

    @objective(model, Min, sum(w[i, k] * psi[i, k] for (i, k) in edges))

    out = zeros(num_obs, cust_dim)
    for ins in 1:num_obs
        for i in 1:cust_dim
            set_normalized_rhs(con_class[i], -states[ins, i])
        end
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            error("interval $n instance $ins: $(termination_status(model))")
        end
        out[ins, :] = drift(value.(psi))
    end
    return out
end

w = reference_weights(policy)

for n in 0:(num_intervals - 1)
    vals = solve_interval(n, w)
    CSV.write(joinpath(out_dir, @sprintf("d_term_output_%s_%03d.csv", policy, n)),
              DataFrame(vals, :auto); writeheader = false)
    println("interval $n done, mean |D| = ", sum(abs.(vals)) / length(vals))
end
