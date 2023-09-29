

import copy
import queue
import threading
from data_loader.data_loader_service import DataLoaderService
from model_service.model_service import ModelService
from models.neural_network_metrics import NNMetrics
from neural_networks.deep_mlp import DeepMLP
from neural_networks.multi_layer_perceptron import MLP
from neural_networks.perceptron import Perceptron
from data_loader.torch_data import TorchData
from plot_service.plot_service import PlotService
from utils.utils import Utils
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader


def train_model(model_name, epochs, file_name, model, criterion, optimiser, device, batch_size):
    model_service = ModelService(
        num_epoch=epochs,
        file_path=file_name,
        model=model,
        criterion_func=criterion,
        optimiser_func=optimiser,
        save_file_name=f"{model_name}_best_model.pt",
        performance_metric="loss",
        save_model=True,
        training_size=0.7,
        validation_size=0.15,
        device=device,
        batch_size=batch_size
    )

    # Start program
    train_performance, valid_performance = model_service.start_model()

    return valid_performance


def worker_function(model_name, epochs, file_name, model, criterion, optimiser, device, batch_size, queue):
    # Perform your algorithm computations
    # Add the results to the queue
    print(f"Creating Worker {model_name}\n")
    results: NNMetrics = train_model(
        model_name, epochs, file_name, model, criterion, optimiser, device, batch_size)
    queue.put(results)


def GridSearch():
    plot_service = PlotService()

    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )

    device = "cpu"

    file_name = "diabetes_pre_processed.txt"
    num_features = 8
    epochs = 300
    batch_size = 100

    perceptron = Perceptron(num_features, "Perceptron")
    mlp_1 = MLP(num_features, "MLP_16")
    mlp_16 = copy.deepcopy(mlp_1)
    mlp_32 = MLP(num_features, "MLP_32", 32)
    dmlp_32 = DeepMLP(num_features, "DeepMLP_32", [16, 32, 16])
    dmlp_64 = DeepMLP(num_features, "DeepMLP_64", [32, 64, 32])

    initial_lr = 0.1

    optimiser = torch.optim.SGD(perceptron.parameters(), lr=initial_lr)

    # Binary classification
    criterion = nn.BCEWithLogitsLoss()

    result_queue = queue.Queue()

    thread1 = threading.Thread(target=worker_function, args=(
        perceptron.name,
        epochs,
        file_name,
        perceptron,
        criterion,
        optimiser,
        device,
        batch_size,
        result_queue,
    ))
    thread2 = threading.Thread(target=worker_function, args=(
        mlp_16.name,
        epochs,
        file_name,
        mlp_16,
        criterion,
        optimiser,
        device,
        batch_size,
        result_queue,
    ))
    thread3 = threading.Thread(target=worker_function, args=(
        mlp_32.name,
        epochs,
        file_name,
        mlp_32,
        criterion,
        optimiser,
        device,
        batch_size,
        result_queue,
    ))
    thread4 = threading.Thread(target=worker_function, args=(
        dmlp_32.name,
        epochs,
        file_name,
        dmlp_32,
        criterion,
        optimiser,
        device,
        batch_size,
        result_queue,
    ))
    thread5 = threading.Thread(target=worker_function, args=(
        dmlp_64.name,
        epochs,
        file_name,
        dmlp_64,
        criterion,
        optimiser,
        device,
        batch_size,
        result_queue,
    ))

    # Initialise in threats
    # thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()

    # thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()

    # Collect and process the results from the queue
    data = []
    while not result_queue.empty():
        results: NNMetrics = result_queue.get()
        data.append(results)

    models = {}
    for model in data:
        models[model.model_name] = model

    # plot_service.plot_epochs(
    #     len(data[0].get_metric("loss")), models, f"AllModelsLoss_epochs_{epochs}_batch_{batch_size}_lr_{initial_lr}", ["loss"])
    
    plot_service.plot_epochs(
        len(data[0].get_metric("accuracy")), models, f"AllModelsAccuracy_epochs_{epochs}_batch_{batch_size}_lr_{initial_lr}", ["accuracy"])
    
    plot_service.plot_epochs(
        len(data[0].get_metric("precision")), models, f"AllModelsPrecision_epochs_{epochs}_batch_{batch_size}_lr_{initial_lr}", ["precision"])
    

    plot_service.plot_epochs(
        len(data[0].get_metric("recall")), models, f"AllModelsRecall_epochs_{epochs}_batch_{batch_size}_lr_{initial_lr}", ["recall"])
    
    # plot_service.plot_epochs(
    #     epochs, models, f"AllModelsLearningRate_epochs_{epochs}_batch_{batch_size}_lr_{initial_lr}", ["learning_rate"])

    return


def run_all():
    file_name = "diabetes_pre_processed.txt"
    num_features = 8

    # Set models
    perceptron = Perceptron(num_features)
    mlp_16 = MLP(num_features)
    mlp_32 = MLP(num_features, 32)
    dmlp_32 = DeepMLP(num_features, [16, 32, 16])
    dmlp_64 = DeepMLP(num_features, [32, 64, 32])

    models = [perceptron, mlp_16, mlp_32, dmlp_32, dmlp_64]

    # Batches
    batches = [5, 10, 50, 100]

    # Epoch
    epoch = [100, 200, 500]

    return


def main():

    plot_service = PlotService()
    file_name = "diabetes_pre_processed.txt"
    num_features = 8

    epochs = 100

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # model = DeepMLP(num_features)
    model = MLP(num_features)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

    # Binary classification
    criterion = nn.BCEWithLogitsLoss()

    # Model service
    model_service = ModelService(
        num_epoch=epochs,
        file_path=file_name,
        model=model,
        criterion_func=criterion,
        optimiser_func=optimiser,
        save_file_name="best_model.pt",
        performance_metric="loss",
        save_model=True,
        training_size=0.7,
        validation_size=0.15,
        device=device,
        batch_size=10
    )

    # Start program
    train_performance, valid_performance = model_service.start_model()

    metric = {
        "MLP": valid_performance
    }

    plot_service.plot_epochs(
        epochs, metric, "Model Metrics", ["loss", "accuracy"])

    return


def main2():
    file_name = "diabetes_pre_processed.txt"
    num_features = 8
    epochs = 500
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
    epochs = 500
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


def plot_data():
    file_name = "diabetes.csv"
    target_feature = "Outcome"

    data_service = DataLoaderService()
    plot_service = PlotService()

    df = data_service.load_csv(file_name)

    # Plot distribution plot
    plot_service.plot_data_distribution(df, target_feature)


if __name__ == "__main__":
    # plot_data()
    # main()
    GridSearch()
