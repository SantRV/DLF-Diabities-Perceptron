import torch.nn as nn
from perceptron.torch_data import TorchData
from torch.utils.data import DataLoader
from utils.utils import Utils


class Perceptron(nn.Module):
    utils = Utils()
    data = None

    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)  # Updated to have 1 output unit
        # self.sigmoid = nn.Sigmoid()

    def load_data(self, file_name: str, batch_size=5, is_training=True):
        df = self.utils.load_data_txt(file_name)
        self.data = df
        dataset = TorchData(df)
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        return self.output_layer(x)
        # return self.sigmoid(logits)

    def get_data(self):
        return self.data
