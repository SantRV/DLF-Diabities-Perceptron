from torch.utils.data import DataLoader, random_split
from data_loader.torch_data import TorchData
from utils.utils import Utils
import pandas as pd


class DataLoaderService():
    def __init__(self) -> None:
        self.utils = Utils()
        pass

    def load_csv(self, file_name: str):
        return self.utils.load_data_csv(file_name)

    def __load_data_from_file(self, file_name: str):
        return self.utils.load_data_txt(file_name)

    def __load_data(self, file_name: str, batch_size=5, is_training=True):
        df = self.utils.load_data_txt(file_name)
        dataset = TorchData(df)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=is_training)
        self.data = dataloader  # Store the DataLoader directly
        return dataloader

    def get_data(self, file_name: str, batch_size=5, is_training=True, training_size: float = 0.7, validation_size: float = 0.15):
        # Validate input
        if (training_size + validation_size >= 1):
            raise ValueError(
                "Invalid data size, training and validation cannot exceed 100%")

        # Load data set
        self.__load_data(file_name)  # Use the DataLoader stored in self.data

        # Define data partitions
        # Access dataset from DataLoader
        train_size = int(training_size * len(self.data.dataset))
        val_size = int(validation_size * len(self.data.dataset))
        test_size = len(self.data.dataset) - train_size - val_size

        # Use random_split to create the splits
        train_dataset, val_dataset, test_dataset = random_split(
            self.data.dataset, [train_size, val_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=is_training)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader
