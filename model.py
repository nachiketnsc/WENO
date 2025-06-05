# model.py
import torch
import torch.nn as nn
import torch.nn.init as init
from config import device

class PINN(nn.Module):
    def __init__(self, layers: list[int]):
        """
        layers: e.g. [2, 50, 50, 50, 1]
          – 2 inputs (x,t)
          – 3 hidden layers of 50 neurons each
          – 1 output
        """
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.net = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))

        # Weight initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x, t: each [N,1]
        → Concatenate → feed through MLP → return u_pred [N,1]
        """
        inp = torch.cat([x, t], dim=1)  # [N,2]
        h = inp
        for i in range(len(self.net) - 1):
            h = self.activation(self.net[i](h))
        u_out = self.net[-1](h)  # last layer, no activation
        return u_out


class WPINN(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_hidden_layers: int,
                 hidden_neurons: int,
                 family_size: int):
        """
        Example from your prior Model.py (for “W‐PINN”):
          – input_size: typically 1 (since you approximate coefficients vs x)
          – num_hidden_layers: e.g. 3
          – hidden_neurons: e.g. 50
          – family_size: how many basis functions you output per x
        """
        super(WPINN, self).__init__()
        self.activation = nn.Tanh()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(self.activation)

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)

        # Output → family_size
        layers.append(nn.Linear(hidden_neurons, family_size))
        self.network = nn.Sequential(*layers)

        # Weight init
        for m in self.network:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

        # Learnable scalar bias
        self.bias = nn.Parameter(torch.tensor(0.5, dtype=torch.float32,
                                              device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [N,1]
        inp = x.view(-1, 1)
        coeffs = self.network(inp)     # [N, family_size]
        return coeffs, self.bias
