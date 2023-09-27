import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=16):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.batchnorm = nn.BatchNorm1d(hidden_size)  # Add batch normalization

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)
