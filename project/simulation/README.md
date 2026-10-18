# Simulation

Discrete event simulation of the transfer network under the index rules of
the paper, on the test problem.  Self contained, it depends on the test
problem parameters written by `../pipeline/` and on the code here.

    python make_inputs.py      # calibration CSV files -> data/ and config_<rule>.json
    ./build.sh                 # Gurobi build -> ./transfer_sim
    ./run.sh                   # every rule, replications 0..999 -> results/<rule>.csv
    ./run.sh 0 100 fsf value   # replications 0..99 of two rules

## Index rules

Each attaches a weight $w_j$ to activity $j$.  At every event the linear
program maximizes the weighted number of callers in service over the
feasible actions, callers in the system per class and agents on duty per
pool, with every caller in the system reassignable.

| name | weight $w_j$ | paper |
|---|---|---|
| `cmu` | $c_{b(j)}\mu_j$ | the $c\mu$ rule |
| `cmu_theta` | $c_{b(j)}\mu_j/\theta_{b(j)}$ | the $c\mu/\theta$ rule |
| `fsf` | $\mu_j$ | fastest server first |
| `myopic` | $c_{b(j)} + \bar v_j\mu_j$ | the myopic rule |
| `myopic_theta` | $(c_{b(j)} + \bar v_j\mu_j)/\theta_{b(j)}$ | abandonment adjusted myopic |
| `value` | $\bar v_j\mu_j$ | the value rate rule |
| `prices` | $w_j(p)$ below with $p(t)$ read from `prices_path`, one row per grid interval | a price table, the fluid price rule when $p = -y$ with $y$ the planning problem's class duals (`../inputs/make_fluid_rule.py`) |

Replications run in parallel with OpenMP, each with its own Gurobi model on
one thread.

## The trained policy

Both simulators also run the policy of the deep splitting solver, `policy`
`nn` in the config written as `config_nn.json`, and policy `prices` runs the
same weight formula with a fixed price table in place of the gradient.  `nn_policy.h` holds the
gradient networks, read from the text weights the solver writes
(`nn_dir`, by default `../neural_network/cppweights_main`),
and the weights of the policy.  At every decision the network of the current
grid point is evaluated at the centered state $\hat x_i = (X_i - x^{nom}_i(t))/\sqrt n$,
with $x^{nom}_i(t) = \sum_j \xi_{ij}(t)N_j(t)$, giving $p = \nabla V$, and activity
$(i,j)$ gets the weight

    w_ij(p) = (mu_ij - theta_i) p_i - mu_ij sum_l P_(ij),l p_l + vbar_ij mu_ij + c_i

the objective of the linear program defining $M$.  `transfer_sim` sets these
as the objective coefficients before every solve.

    ./transfer_sim config_nn.json 0 1000 results/nn.csv
