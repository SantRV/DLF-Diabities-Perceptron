import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size, name):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.name = name

    def forward(self, x):
        x = self.fc1(x)
        return self.sigmoid(x)
