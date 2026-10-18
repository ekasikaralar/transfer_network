import numpy as np
import torch
import time
from torch import nn
import os
import json
import pandas as pd


from Net import Y_Net
from Net import Z_Net
import equation as eqn

torch.manual_seed(73)

class DSPL_solver(object):
    """
    DSPL Solver Class for solving PDE-related systems using neural networks.

    Attributes:
        config (object): General configuration object containing network and equation parameters.
        bsde (object): Object implementing the PDE method (e.g., HJB, FBSDE).
    """

    def __init__(self, config, bsde):

        # Configuration objects
        self.config = config
        self.net_config = config.net_config  # Neural Network Hyperparameters
        self.eqn_config = config.eqn_config  # PDE-related system parameters
        self.bsde = bsde  # PDE method instance

        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

        # PDE dimensions and time parameters
        self.dim = self.eqn_config.dim  # Dimensionality of the PDE
        self.num_time_interval = self.eqn_config.num_time_interval  # Number of time intervals partitioning the time horizon
        # The model grid: zeta, sigma and the offline networks are estimated at
        # five minute resolution and change only on that grid, while the
        # splitting step above can be refined below it.
        self.num_model_interval = getattr(self.eqn_config, "num_model_interval",
                                          self.num_time_interval)
        self.delta_t = self.eqn_config.total_time / self.num_time_interval  # Length of each time partition
        self.sqrt_delta_t = np.sqrt(self.delta_t)  # Square root of time partition length
        self.T = self.eqn_config.total_time  # Total time horizon
        self.print_interval = self.net_config.print_interval

        # Neural Network architecture parameters
        self.num_neurons = self.net_config.num_neurons  # Neurons in each hidden layer
        self.num_layers = self.net_config.num_layers  # Number of hidden layers
        self.activation = self.net_config.activation  # Activation function for hidden layers
        self.slope = self.net_config.slope  # Alpha for Leaky ReLU or CELU activations
        self.valid_size = self.net_config.valid_size
        self.batch_size = self.net_config.batch_size
        self.prev_init = self.net_config.prev_init

        # Learning rate parameters
        self.gamma = self.net_config.gamma  # Decay rate for learning rate
        self.inner_learning_rate = self.net_config.inner_learning_rate  # Initial learning rate for inner intervals
        self.final_learning_rate = self.net_config.final_learning_rate  # Initial learning rate for the final interval
        self.inner_milestones = self.net_config.inner_milestones  # Milestones for adjusting learning rate (inner intervals)
        self.final_milestones = self.net_config.final_milestones  # Milestones for adjusting learning rate (final interval)

        # Gradient clipping parameters
        self.grad_clip = self.net_config.grad_clip  # Enable/disable gradient clipping
        self.grad_norm = self.net_config.grad_norm  # Maximum gradient norm for clipping

        # Early stopping parameters
        self.early_stopping = self.net_config.early_stopping  # Enable/disable early stopping
        self.patience = self.net_config.patience  # Iteration patience for early stopping
        self.min_delta = self.net_config.min_delta  # Minimum improvement for early stopping

        # Load system data (rates, costs, and distributions)
        d = self.eqn_config.data_dir
        rd = lambda name: torch.tensor(
            pd.read_csv(os.path.join(d, name), header=None).values,
            dtype=torch.float32).to(self.device)

        self.theta = rd(self.eqn_config.theta_file).reshape(-1)  # Hourly abandonment rates
        self.cost = rd(self.eqn_config.cost_file).reshape(-1)  # Hourly cost rates
        self.zeta = rd(self.eqn_config.zeta_file)  # Second-order drift terms, num_model_interval x dim
        self.mean = rd(self.eqn_config.means_file)  # Means of sample paths, dim x num_time_interval

        # Covariance of the sample paths, one Cholesky factor per grid point.
        self.chol = rd(self.eqn_config.chols_file).reshape(
            self.num_time_interval, self.dim, self.dim)  # fitted on the splitting grid

        # Diffusion coefficient for the state process.  
        self.sigma = rd(self.eqn_config.sigma_file).reshape(
            self.num_model_interval, self.dim, self.dim)

        self.x0_range = self.eqn_config.initial_range
        self.x0_lo, self.x0_hi = None, None
        if getattr(self.eqn_config, "initial_box", False):
            headroom = rd(self.eqn_config.headroom_file).reshape(-1)
            xi0 = rd(self.eqn_config.xi_file)[0].reshape(self.dim, -1)
            agents0 = rd(self.eqn_config.agents_file)[0]
            x_opt0 = (xi0 * agents0[None, :]).sum(1) / self.eqn_config.n_scale
            sqrt_n = float(np.sqrt(self.eqn_config.n_scale))
            self.x0_lo = -sqrt_n * x_opt0
            self.x0_hi = (headroom - self.eqn_config.n_scale * x_opt0) / sqrt_n

    
        self.lambd_grad = getattr(self.net_config, "lambd_grad", 1.0)


    # Generate sample paths
    def sample(self, num_sample, n):
        """
        Generate sample paths for the PDE system.

        Args:
            num_sample (int): Number of samples to generate.
            n (int): Current time interval index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Current state (x_n) and next state (x_{n+1}).
        """
        # Initial state (x0): uniform on the box, drawn independently per path
        u = torch.rand(num_sample, self.dim).to(self.device)
        if self.x0_lo is not None:
            x0 = self.x0_lo + u * (self.x0_hi - self.x0_lo)
        else:
            x0 = (u * 2 - 1) * self.x0_range

        # Adjust current state based on the fitted distribution if not the
        # first interval, drawing the classes jointly so the correlation the
        # reference process builds up is preserved.
        if n != 0:
            gauss = torch.randn(num_sample, self.dim).to(self.device)
            x0 = self.mean[:, n] + torch.matmul(gauss, self.chol[n].T)

        # Generate random noise (W) for diffusion
        W = torch.randn(num_sample, self.dim).to(self.device) * self.sqrt_delta_t

        # Compute drift and diffusion terms.  
        m = self.bsde.model_index(n)
        with torch.no_grad():
            drift = (self.zeta[m, :] - self.theta * x0 + self.bsde.drift_term(n, x0)) * self.delta_t
            diffu = torch.matmul(W, self.sigma[m].T)

            # Compute the next state
            x1 = x0 + drift + diffu

        return x0, x1


    def loss_fn(self, v_prev, v_next):
        """
        Compute the loss function.

        Args:
            v_prev (torch.Tensor): Predicted value from the previous time step.
            v_next (torch.Tensor): Predicted value from the next time step.

        Returns:
            torch.Tensor: Computed loss value.
        """
        # Difference between consecutive predictions
        delta = v_prev - v_next

        return torch.mean(delta ** 2)

    def step_losses(self, y_net, z_net, n, xn, target):
        """The two parts of the objective at one splitting step.

        The first is the mean squared error of the step itself.  The second is
        the gap between Z and the gradient of Y at the same states.

        Returns the two separately so the logs stay readable and the step error
        remains comparable with earlier runs."""
        xn = xn.detach().requires_grad_(True)

        vn = y_net(xn)
        zn = z_net(xn)
        grad_y = torch.autograd.grad(vn.sum(), xn, create_graph=True)[0]

        rhs = vn + self.bsde.f_tf(n, xn, self.cost, zn) * self.delta_t

        step = self.loss_fn(target, rhs)
        gap = torch.mean(torch.sum((zn - grad_y) ** 2, dim=1))
        return step, gap

    def save_weights_as_txt(self, network, network_name, save_path_cppweights):
        # Iterate through each layer in the network
        for j, layer in enumerate(network.linearlayers):
            # Save weights
            if hasattr(layer, 'weight'):
                weight_filename = os.path.join(save_path_cppweights, f"{network_name}_w{j}.txt")
                np.savetxt(weight_filename, layer.weight.detach().cpu().numpy(), delimiter = ",")
            # Save biases if they are not None
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_filename = os.path.join(save_path_cppweights, f"{network_name}_b{j}.txt")
                np.savetxt(bias_filename, layer.bias.detach().cpu().numpy(), delimiter = ",")

    def train(self, final_iterations, inner_iterations, save_path_logs, save_path_cppweights):

        # Step 1: Initialize neural networks for all intervals
        y_net_list = nn.ModuleList([Y_Net(self.dim, self.num_neurons, self.num_layers, self.activation, self.slope) for _ in range(self.num_time_interval)]).to(self.device) # Network for value function approximation for each interval
        z_net_list = nn.ModuleList([Z_Net(self.dim, self.num_neurons, self.num_layers, self.activation, self.slope) for _ in range(self.num_time_interval)]).to(self.device)  # Network for gradient approximation for each interval

        training_history = []
        start_time = time.time()

        # Initialize optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(
                                    list(y_net_list[self.num_time_interval - 1].parameters()) +
                                    list(z_net_list[self.num_time_interval - 1].parameters()), lr = self.final_learning_rate)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.final_milestones, gamma = self.gamma)

        # Generate validation data for the interval
        xn_valid, xn1_valid = self.sample(self.valid_size, self.num_time_interval - 1)

        # Step 2: Train the network for the final interval
        for i in range(final_iterations):

            # Training step
            y_net_list[self.num_time_interval - 1].train()
            z_net_list[self.num_time_interval - 1].train()

            # Generate training data
            xn_train, xn1_train = self.sample(self.batch_size, self.num_time_interval - 1)

            # Compute target and predictions
            with torch.no_grad():
                vn1_train = self.bsde.g_tf(xn1_train)

            step_train, gap_train = self.step_losses(
                y_net_list[self.num_time_interval - 1],
                z_net_list[self.num_time_interval - 1],
                self.num_time_interval - 1, xn_train, vn1_train)

            loss = step_train + self.lambd_grad * gap_train
            loss_train = step_train.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping if enabled
            if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(list(y_net_list[self.num_time_interval - 1].parameters()) +
                                list(z_net_list[self.num_time_interval - 1].parameters()), self.grad_norm)


            optimizer.step()
            scheduler.step()

            # Validation step.  
            y_net_list[self.num_time_interval - 1].eval()
            z_net_list[self.num_time_interval - 1].eval()

            with torch.no_grad():
                vn1_valid = self.bsde.g_tf(xn1_valid)

            step_valid, gap_valid = self.step_losses(
                y_net_list[self.num_time_interval - 1],
                z_net_list[self.num_time_interval - 1],
                self.num_time_interval - 1, xn_valid, vn1_valid)
            loss_valid = step_valid.item()
            gap_v = gap_valid.item()

            # Log progress
            if i % self.print_interval == 0:
                elapsed_time = time.time() - start_time
                training_history.append([self.num_time_interval - 1, i, loss_train, loss_valid, gap_v, elapsed_time])
                print("interval: ", self.num_time_interval - 1, " iter: ", i, " train_loss: ", loss_train, " validation_loss: ", loss_valid, " grad_gap: ", gap_v, " elapsed_time: ", elapsed_time)

        # Save the trained model for the current interval
        torch.save(y_net_list[self.num_time_interval - 1].state_dict(), save_path_logs + f"y_network{self.num_time_interval - 1}.pth")
        torch.save(z_net_list[self.num_time_interval - 1].state_dict(), save_path_logs + f"z_network{self.num_time_interval - 1}.pth")

        # Step 3: Train networks for the remaining intervals in reverse order
        for n in range(self.num_time_interval - 2, -1, -1):

            # Load weights from the next interval and freeze them
            if self.prev_init:
                y_net_list[n].load_state_dict(y_net_list[n+1].state_dict())
                z_net_list[n].load_state_dict(z_net_list[n+1].state_dict())

            for param in y_net_list[n+1].parameters():
                param.requires_grad = False

            for param in z_net_list[n+1].parameters():
                param.requires_grad = False

            # Initialize optimizer and learning rate scheduler
            optimizer = torch.optim.Adam(
                                        list(y_net_list[n].parameters()) +
                                        list(z_net_list[n].parameters()), lr = self.inner_learning_rate)

            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.inner_milestones, gamma = self.gamma)

            # Generate validation data for the interval
            xn_valid, xn1_valid = self.sample(self.valid_size, n)

            best_val_loss = float('inf') # Track the best validation loss
            patience_counter = 0 # Counter for early stopping
            patience = self.patience  # Number of iterations with no improvement, after which training will be stopped
            min_delta = self.min_delta

            for i in range(inner_iterations):

                # Training step
                y_net_list[n].train()
                z_net_list[n].train()

                # Generate training data
                xn_train, xn1_train = self.sample(self.batch_size, n)

                # Compute target and predictions
                with torch.no_grad():
                    vn1_train = y_net_list[n+1](xn1_train)

                step_train, gap_train = self.step_losses(
                    y_net_list[n], z_net_list[n], n, xn_train, vn1_train)

                loss = step_train + self.lambd_grad * gap_train
                loss_train = step_train.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping if enabled
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(list(y_net_list[n].parameters()) +
                                list(z_net_list[n].parameters()), self.grad_norm)

                optimizer.step()
                scheduler.step()

                # Validation step.  The gradient gap needs autograd through Y,
                # so this cannot sit inside no_grad, but nothing is optimized.
                y_net_list[n].eval()
                z_net_list[n].eval()

                with torch.no_grad():
                    vn1_valid = y_net_list[n+1](xn1_valid)

                step_valid, gap_valid = self.step_losses(
                    y_net_list[n], z_net_list[n], n, xn_valid, vn1_valid)
                loss_valid = step_valid.item()
                gap_v = gap_valid.item()

                # Handle early stopping
                if self.early_stopping:
                    if loss_valid < best_val_loss - min_delta:
                        best_val_loss = loss_valid
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter > patience:
                        print(f"Early stopping at iteration {i}, time interval {n}")
                        break

                # Log progress
                if i % self.print_interval == 0:
                    elapsed_time = time.time() - start_time
                    training_history.append([n, i, loss_train, loss_valid, gap_v, elapsed_time])
                    print("interval: ", n, " iter: ", i, " train_loss: ", loss_train, " validation_loss: ", loss_valid, " grad_gap: ", gap_v, " elapsed_time: ", elapsed_time)

            # Save the trained model for the current interval
            torch.save(y_net_list[n].state_dict(), save_path_logs + f"y_network{n}.pth")
            torch.save(z_net_list[n].state_dict(), save_path_logs + f"z_network{n}.pth")

        # Step 4: Save the final weights for use in C++ simulation
        for n in range(self.num_time_interval):
            self.save_weights_as_txt(z_net_list[n], f"z_network{n}", save_path_cppweights)

        return training_history
