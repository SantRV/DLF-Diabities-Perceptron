import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size, name):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.name = name

    def forward(self, x):
        return self.fc1(x)
