import torch
from torch.utils.data import Dataset


class TorchData(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Exclude the first column (target)
        features = torch.FloatTensor(self.data.iloc[idx, 1:].values)
        # Map -1 to 0 and 1 to 1
        target = torch.FloatTensor([0 if self.data.iloc[idx, 0] == -1 else 1])

        return features, target
