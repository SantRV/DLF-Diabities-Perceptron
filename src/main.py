

from neural_networks.perceptron import Perceptron
from data_loader.torch_data import TorchData
from utils.utils import Utils
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader


def predict(model, data):
    model.eval()
    # model is self(VGG class's object)

    return model(data)


def train(dataloader, model, loss_fn, optimiser, epoch, epochs):
    for batch in dataloader:
        features, target = batch
        optimiser.zero_grad()
        output = model(features)
        loss = loss_fn(output, target.view(-1))
        loss.backward()
        optimiser.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

    # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

    # for batch in dataloader:
    #     features, target = batch
    #     X, y = features.to(device), target.to(device)

    #     # Prediction
    #     pred = model(X)
    #     loss = loss_fn(pred, y)

    #     # Backpropagation
    #     loss.backward()
    #     optimiser.step()
    #     optimiser.zero_grad()

    #     loss, current = loss.item(), (i + 1) * len(X)
    #     print(f"loss: {loss:>7f} [{current}/{size}]")
    #     i += 1


def main():
    file_name = "diabetes_pre_processed.txt"
    num_features = 8

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = Perceptron(num_features).to(device)

    dataloader = model.load_data(file_name, batch_size=10)

    # data = pd.DataFrame({'target': [1, -1, 1],
    #                      'feature1': [1.0, 2.0, 3.0],
    #                      'feature2': [4.0, 5.0, 6.0],
    #                      'feature3': [7.0, 8.0, 9.0],
    #                      'feature4': [10.0, 11.0, 12.0],
    #                      'feature5': [13.0, 14.0, 15.0],
    #                      'feature6': [16.0, 17.0, 18.0],
    #                      'feature7': [19.0, 20.0, 21.0],
    #                      'feature8': [22.0, 23.0, 24.0]})

    # batch_size = 2
    # dataset = TorchData(data)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Binary classification
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    epochs = 100
    testData = None
    best_model = None
    best_loss = float('inf')

    print("Device ", device)
    for epoch in range(epochs):
        for batch in dataloader:
            if testData is None:
                testData = batch
            features, target = batch
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output.view(-1), target.view(-1))

            if loss < best_loss:
                best_loss = loss
                best_model = model

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

    # Predictions
    data = model.get_data()
    features = torch.FloatTensor(data.iloc[0, 1:].values)
    target = torch.FloatTensor([0 if data.iloc[0, 0] == -1 else 1])

    predictions = predict(best_model, features.unsqueeze(0))

    loss = criterion(predictions, target.unsqueeze(0))

    # Threshold the predictions to get binary values (0 or 1)
    binary_predictions = (predictions >= 0.5).float()

    print(f"Min Loss: {best_loss}")
    print(f"Y: {target.view(-1)} Prediction: {predictions.view(-1)} Loss {loss.item()} Pred: {binary_predictions}")


if __name__ == "__main__":
    main()
