import numpy as np
import torch
from torch import nn


class Y_Net(nn.Module):

    def __init__(self, dim, units, num_layers, activation, slope):
        super(Y_Net, self).__init__()

        self.linearlayers = nn.ModuleList([nn.Linear(dim, units)])

        for _ in range(num_layers - 1):  # num_layers is the number of hidden layers
            self.linearlayers.append(nn.Linear(units, units))

        self.linearlayers.append(nn.Linear(units, 1, bias=False))

        if activation == 'ReLU':
            self.activation = nn.ReLU()
            init_method = nn.init.kaiming_uniform_

        elif activation == 'eLU':
            self.activation = nn.ELU()
            init_method = nn.init.kaiming_uniform_

        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope = slope)
            init_method = nn.init.kaiming_uniform_

        elif activation == "GELU":
            self.activation = nn.GELU()
            init_method = nn.init.xavier_uniform_

        elif activation == "SELU":
            self.activation = nn.SELU()
            init_method = nn.init.xavier_uniform_

        elif activation == "SiLU":
            self.activation = nn.SiLU()
            init_method = nn.init.xavier_uniform_

        elif activation == "CELU":
            self.activation = nn.CELU(alpha=slope)
            init_method = nn.init.xavier_uniform_

        else:
            raise Exception(f"{activation} not an available activation. Available options: ReLU, eLU, LeakyReLU, GELU, SELU, SiLU, CELU")

        for layer in self.linearlayers:
            if isinstance(layer, nn.Linear):
                init_method(layer.weight)


    def forward(self, x):

        # hidden layers
        for layer in self.linearlayers[:-1]:
            x = self.activation(layer(x))

        # pass through the output layer
        x = self.linearlayers[-1](x)

        return x



class Z_Net(nn.Module):

    def __init__(self, dim, units, num_layers, activation, slope):
        super(Z_Net, self).__init__()

        self.linearlayers = nn.ModuleList([nn.Linear(dim, units)])

        for _ in range(num_layers - 1):
            self.linearlayers.append(nn.Linear(units, units))

        self.linearlayers.append(nn.Linear(units, dim, bias=False))

        if activation == 'ReLU':
            self.activation = nn.ReLU()
            init_method = nn.init.kaiming_uniform_

        elif activation == 'eLU':
            self.activation = nn.ELU()
            init_method = nn.init.kaiming_uniform_

        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope = slope)
            init_method = nn.init.kaiming_uniform_

        elif activation == "GELU":
            self.activation = nn.GELU()
            init_method = nn.init.xavier_uniform_

        elif activation == "SELU":
            self.activation = nn.SELU()
            init_method = nn.init.xavier_uniform_

        elif activation == "SiLU":
            self.activation = nn.SiLU()
            init_method = nn.init.xavier_uniform_

        elif activation == "CELU":
            self.activation = nn.CELU(alpha=slope)
            init_method = nn.init.xavier_uniform_
        else:
            raise Exception(f"{activation} not an available activation. Available options: ReLU, eLU, LeakyReLU, GELU, SELU, SiLU, CELU")

        for layer in self.linearlayers:
            if isinstance(layer, nn.Linear):
                init_method(layer.weight)


    def forward(self, x):

        # hidden layers
        for layer in self.linearlayers[:-1]:
            x = self.activation(layer(x))

        # pass through the output layer
        x = self.linearlayers[-1](x)

        return x
