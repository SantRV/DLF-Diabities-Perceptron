from torch.utils.data import DataLoader, random_split
from data_loader.torch_data import TorchData
from utils.utils import Utils


class DataLoaderService():
    def __init__(self) -> None:
        self.utils = Utils()
        pass

    def __load_data_from_file(self, file_name: str):
        return self.utils.load_data_txt(file_name)

    def __load_data(self, file_name: str, batch_size=5, is_training=True):
        df = self.utils.load_data_txt(file_name)
        self.data = df
        dataset = TorchData(df)
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)

    def get_data(self, file_name: str, batch_size=5, is_training=True, training_size: float = 0.7, validation_size: float = 0.15):
        # Validate input
        if (training_size + validation_size >= 1):
            raise ValueError

        # Load data set
        dataset = self.__load_data_from_file(file_name)
        if len(dataset == 0):
            raise ValueError

        # Define data partitions
        train_size = int(training_size * len(dataset))
        val_size = int(validation_size * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Use random_split to create the splits
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=is_training)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader
