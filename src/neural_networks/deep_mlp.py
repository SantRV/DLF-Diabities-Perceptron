import torch.nn as nn


class DeepMLP(nn.Module):
    def __init__(self, input_size, name, hidden_sizes=[16, 32, 16], output_size=1):
        super(DeepMLP, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.Sigmoid())

        self.name = name

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
