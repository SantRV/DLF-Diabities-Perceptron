
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from data_loader.data_loader_service import DataLoaderService


class ModelService():
    def __init__(self,
                 num_epoch: int,
                 file_path: str,
                 model,
                 criterion_func,
                 optimiser_func,
                 save_file_name: str,
                 performance_metric: str = "loss",
                 save_model: bool = True,
                 training_size: float = 0.7,
                 validation_size: float = 0.15,
                 device: str = "cpu",
                 batch_size: int = 5,
                 ) -> None:

        self.num_epoch: int = num_epoch
        self.file_path: str = file_path
        self.model = model
        self.criterion_func = criterion_func
        self.optimiser_func = optimiser_func
        self.performance_metric = performance_metric
        self.save_file_name: str = save_file_name
        self.save_model: bool = save_model
        self.training_size: float = training_size
        self.validation_size: float = validation_size
        self.device: str = device
        self.batch_size: int = batch_size

        # Utils
        self.data_service = DataLoaderService()

        pass

    def train_model_one_epoch(self, model, epoch: int, dataloader):
        # Set to training mode
        model.train()

        running_loss = 0.0

        # Use enumerate to get the batch index
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero parameter gradients
            self.optimiser_func.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Loss computation
            loss = self.criterion_func(outputs, labels)

            # Backpropagation
            loss.backward()

            # Optimization step
            self.optimiser_func.step()

            # Accumulate loss
            running_loss += loss.item()

            # Print statistics every 2000 minibatches
            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}, Batch {i + 1}] Loss: {running_loss / 50:.3f}")
                running_loss = 0.0

        return running_loss

    def comp_accuracy(self, outputs, labels):
        """
        Accuracy is a measure of how many predictions made by the model are correct out of the total predictions
        Accuracy = (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)

        True Positives (TP): The number of instances correctly predicted as positive.
        True Negatives (TN): The number of instances correctly predicted as negative.
        False Positives (FP): The number of instances incorrectly predicted as positive.
        False Negatives (FN): The number of instances incorrectly predicted as negative.
        """

        return accuracy_score(outputs, labels)

    def comp_precision(self, outputs, labels):
        """
        Precision is a measure of how many of the positive predictions made by the model were actually correct.
        It is calculated as the ratio of True Positives to the sum of True Positives and False Positives.

        Precision = TP / (TP + FP)
        """

        return precision_score(outputs, labels, zero_division=0)

    def comp_recall(self, outputs, labels):
        """
        Recall measures how many of the actual positive instances were correctly predicted by the model.
        It is calculated as the ratio of True Positives to the sum of True Positives and False Negatives.

        Recall = TP / (TP + FN)
        """

        return recall_score(outputs, labels, zero_division=0)

    def evaluate_model(self, outputs, labels):
        loss = self.criterion_func(outputs, labels).item()

        # Convert PyTorch tensors to NumPy arrays using .detach().numpy()
        output_np = outputs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        accuracies = []
        precisions = []
        recalls = []

        # Loop through batch elements

        accuracy = self.comp_accuracy(
            output_np, labels_np)
        precision = self.comp_precision(
            output_np, labels_np)
        recall = self.comp_recall(
            output_np, labels_np)

        # accuracies.append(accuracy)
        # precisions.append(precision)
        # recalls.append(recall)

        # # Compute the mean values
        # mean_accuracy = np.mean(accuracies)
        # mean_precision = np.mean(precisions)
        # mean_recall = np.mean(recalls)

        return loss, accuracy, precision, recall

    def validate_model(self, model, dataloader):
        results = {
            "loss": [],
            "accuracy": [],
            "recall": [],
            "precision": []
        }

        # Set to evaluation mode
        model.eval()

        # Validate on data
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Predictions -> Forward pass
            prediction = model(inputs)

            # Parse predictions
            binary_predictions = (prediction >= 0.5).float()

            closs, accuracy, recall, precision = self.evaluate_model(
                binary_predictions, labels)

            # Update metrics
            results["loss"] = closs
            results["accuracy"] = accuracy
            results["precision"] = precision
            results["recall"] = recall

        return results

    def __compare_performance(self, current_best, performance):
        if current_best == None:
            return True

        if performance[self.performance_metric] < current_best:
            return True

        return False

    def train_model(self, trainloader, validloader):
        initial_lr = 0.1
        train_performance = []
        valid_performance = []
        vaccuracies = []
        best_valid_performance = None

        last_converge = 0
        num_convergence = 1
        for epoch in range(self.num_epoch):
            # Train model and get running loss
            ctrain_perform = self.train_model_one_epoch(
                self.model, epoch, trainloader)
            train_performance.append(ctrain_perform)

            # Check performance on validatation data
            cvalid_perform = self.validate_model(self.model, validloader)
            valid_performance.append(cvalid_perform)

            vaccuracies.append(cvalid_perform["accuracy"])

            # Save if best model
            if self.__compare_performance(best_valid_performance, cvalid_perform):
                # Update current best
                best_valid_performance = cvalid_perform[self.performance_metric]

                if self.save_model:
                    torch.save(
                        self.model.state_dict(), f"{self.save_file_name}")

                # Best model
                print(
                    f"=> Best Model {self.performance_metric}: {cvalid_perform[self.performance_metric]} {cvalid_perform}")

                # Adjust learning rate
                for param_group in self.optimiser_func.param_groups:
                    param_group['lr'] = (initial_lr) / (num_convergence)
                    print(
                        f"=>[[Convergence {num_convergence}] Learning Rate {(initial_lr) / (num_convergence)} ")

                    initial_lr = (initial_lr) / (num_convergence)

                    num_convergence += 1

            else:
                # Reduce learning rate linearly
                # Adjust learning rate
                for param_group in self.optimiser_func.param_groups:
                    initial_lr = initial_lr * 0.9
                    param_group['lr'] = initial_lr

            # if last_converge > 20:
            #     # Adjust learning rate
            #     for param_group in self.optimiser_func.param_groups:
            #         param_group['lr'] = (initial_lr) / num_convergence + 1
            #         print(
            #             f"=>[[Convergence {num_convergence}] Learning Rate {(initial_lr ) / num_convergence + 1} ")

            #         initial_lr = (initial_lr) / num_convergence + 1

            #         num_convergence += 1

            if last_converge > 100:

                # Adjust learning rate
                initial_lr = 2
                for param_group in self.optimiser_func.param_groups:
                    param_group['lr'] = initial_lr
                print(
                    f"=>[[Convergence {num_convergence}] Learning Rate {initial_lr} ")

                last_converge = 0

            last_converge += 1

        print(
            f"=> Final Best Model {self.performance_metric}: {best_valid_performance} {cvalid_perform}")
        print(f"Mean accuracy {np.mean(vaccuracies)}")

        # Get accuracy of best model
        best_model = self.model
        best_model.load_state_dict(torch.load(self.save_file_name))
        best_results = {
            "loss": [],
            "accuracy": [],
            "recall": [],
            "precision": []
        }
        for epoch in range(self.num_epoch):
            cvalid_perform = self.validate_model(best_model, validloader)
            best_results["loss"].append(cvalid_perform["loss"])
            best_results["accuracy"].append(cvalid_perform["accuracy"])
            best_results["recall"].append(cvalid_perform["recall"])
            best_results["precision"].append(cvalid_perform["precision"])

            print(
                f"[Epoch {epoch}] Loss {np.mean(cvalid_perform['loss'])} Accuracy {np.mean(cvalid_perform['accuracy'])}")

        print(
            f"==> Best Performance: Loss {np.mean(best_results['loss'])} Accuracy {np.mean(best_results['accuracy'])}")
        return train_performance, valid_performance

    def start(self):
        # Load data
        train_loader, val_loader, test_loader = self.data_service.get_data(
            self.file_path, self.batch_size, True, self.training_size, self.validation_size)

        # Train model
        train_performance, valid_performance = self.train_model(
            train_loader, val_loader)

        # Print Results
        print("--- Training Performance ----")
        print(train_performance)

        print("--- Validation Performance ----")
        print(valid_performance)
