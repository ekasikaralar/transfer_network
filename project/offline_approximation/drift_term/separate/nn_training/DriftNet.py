"""Network that approximates the function D at one grid point.

The input is a state x of dimension I.  The output is the drift vector D(x),
also of dimension I.

One network is trained per grid point t_n, because the nominal allocation
psi*(t_n) and the partition of activities into basic and nonbasic both enter
the reference linear program and both move with t.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

torch.manual_seed(73)


class DriftNet(object):
    def __init__(self, config, interval):
        self.config = config
        self.net_config = config.net_config
        self.data_config = config.data_config
        self.interval = interval

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        d = self.data_config.obs_dir
        rd = lambda name: torch.tensor(
            pd.read_csv(os.path.join(d, name % interval), header=None).values,
            dtype=torch.float32).to(self.device)

        self.state = rd(self.data_config.state_file)
        self.drift = rd(self.data_config.drift_file)

        self.input_dim = self.state.size(1)
        self.output_dim = self.drift.size(1)

        self.model = FeedForwardNet(config, self.input_dim, self.output_dim).to(self.device)

        self.init_learning_rate = self.net_config.init_learning
        self.epochs = self.net_config.epochs
        self.print_interval = self.net_config.print_interval

    def train(self, save_path_logs):

        training_history = []
        start_time = time.time()
        batch_size = self.net_config.batch_size

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.net_config.lr_decay_factor,
            patience=self.net_config.patience)
        loss_fn = nn.MSELoss()

        x = self.state.float()
        y = self.drift.float()

        n_total = x.size(0)
        train_size = int(0.8 * n_total)
        val_size = n_total - train_size
        order = torch.randperm(n_total, device=self.device)
        train_idx, val_idx = order[:train_size], order[train_size:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        for epoch in range(self.epochs):

            self.model.train()
            train_loss = 0.0

            perm = torch.randperm(train_size, device=self.device)
            for i in range(0, train_size, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = x_train[idx], y_train[idx]
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            val_loss = 0.0
            with torch.no_grad():
                self.model.eval()
                for i in range(0, val_size, batch_size):
                    xb, yb = x_val[i:i + batch_size], y_val[i:i + batch_size]
                    loss = loss_fn(self.model(xb), yb)
                    val_loss += loss.item() * xb.size(0)

            train_loss /= train_size
            val_loss /= val_size
            scheduler.step(val_loss)

            if epoch % self.print_interval == 0:
                elapsed_time = time.time() - start_time
                training_history.append([epoch, train_loss, val_loss, elapsed_time])
                print(f"  interval {self.interval:3d}  epoch {epoch:5d}"
                      f"  train_loss {train_loss:.6f}  validation_loss {val_loss:.6f}"
                      f"  elapsed {elapsed_time:.1f}s")

        os.makedirs(save_path_logs, exist_ok=True)
        torch.save(self.model,
                   os.path.join(save_path_logs, f"drift_network_{self.interval:03d}.pth"))
        return training_history


class FeedForwardNet(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super(FeedForwardNet, self).__init__()
        self.net_config = config.net_config

        units = self.net_config.num_neurons
        num_layers = self.net_config.num_layers
        activation = self.net_config.activation

        self.input_batchnorm = nn.BatchNorm1d(input_dim)

        self.linearlayers = nn.ModuleList([nn.Linear(input_dim, units)])
        self.batchnormlayers = nn.ModuleList([nn.BatchNorm1d(units)])

        for _ in range(num_layers - 1):
            self.linearlayers.append(nn.Linear(units, units))
            self.batchnormlayers.append(nn.BatchNorm1d(units))

        self.output_layer = nn.Linear(units, output_dim, bias=False)

        if activation == 'ReLU':
            self.activation = nn.ReLU()
            init_method = nn.init.kaiming_uniform_
        elif activation == 'eLU':
            self.activation = nn.ELU()
            init_method = nn.init.kaiming_uniform_
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
            init_method = nn.init.kaiming_uniform_
        elif activation == 'GELU':
            self.activation = nn.GELU()
            init_method = nn.init.xavier_uniform_
        elif activation == 'SELU':
            self.activation = nn.SELU()
            init_method = nn.init.xavier_uniform_
        elif activation == 'SiLU':
            self.activation = nn.SiLU()
            init_method = nn.init.xavier_uniform_
        elif activation == 'CELU':
            self.activation = nn.CELU()
            init_method = nn.init.xavier_uniform_
        else:
            raise Exception(f"{activation} not an available activation. Available "
                            "options: ReLU, eLU, LeakyReLU, GELU, SELU, SiLU, CELU")

        for layer in self.linearlayers:
            if isinstance(layer, nn.Linear):
                init_method(layer.weight)
        init_method(self.output_layer.weight)

    def forward(self, x):
        x = self.input_batchnorm(x)
        for linear, bn in zip(self.linearlayers, self.batchnormlayers):
            x = self.activation(bn(linear(x)))
        x = self.output_layer(x)
        return x
