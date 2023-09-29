import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from models.neural_network_metrics import NNMetrics


class PlotService():
    def __init__(self) -> None:
        pass

    def plot_data_distribution(self, input_data, target_feature):
        # Create a histogram using Seaborn
        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")
        sns.countplot(x=target_feature, data=input_data, palette="Set3")

        # Add labels and title
        plt.xlabel(target_feature)
        plt.ylabel("Count")
        plt.title(f"Distribution of {target_feature}")

        # Show the plot
        plt.show()

    def plot_epochs(self, epoch, model_metrics: dict, title: str, metric_names: []):
        # Epochs list
        epochs = list(range(epoch))

        # Create a line graph for each model's loss
        plt.figure(figsize=(8, 6))

        # Create line for each model
        y_min = float('inf')
        y_max = 0
        for model_name in model_metrics.keys():
            for metric_name in metric_names:
                plt.plot(epochs, model_metrics[model_name].get_metric(metric_name),
                         linestyle='-', label=f"{model_name} - {metric_name}")

                if min(model_metrics[model_name].get_metric(metric_name)) < y_min:
                    y_min = min(
                        model_metrics[model_name].get_metric(metric_name))

                if max(model_metrics[model_name].get_metric(metric_name)) > y_max:
                    y_max = max(
                        model_metrics[model_name].get_metric(metric_name))

        # Add labels and title
        plt.yticks(np.arange(y_min, y_max+0.1, 0.1))
        plt.xlabel('Number of Epochs')
        plt.ylabel(f'Value')
        plt.title(title)

        # Show legend
        plt.legend()

        # Show gridlines
        plt.grid(True)

        # Show the plot
        self.save_plot(f"NN/results/{title}.png")
        plt.show()
        return

    def save_plot(self, file_path: str):
        if not os.path.exists('NN/results'):
            os.makedirs('NN/results')

        # Saving the plot to a file
        plt.savefig(file_path)

    def plot_correlations(self, input_data, target_column=None):
        data = input_data.copy()

        if target_column == None:
            correlation_matrix = data.corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True,
                        cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Heatmap')
            plt.show()
            return

        else:

            # Calculate the correlations between specific columns and the target column
            correlations = data.drop(
                columns=target_column).corrwith(data[target_column])

            # Create a DataFrame for correlations
            correlation_df = pd.DataFrame({'Correlation': correlations})

            # Sort by absolute correlation value (optional)
            correlation_df = correlation_df.abs().sort_values(
                by='Correlation', ascending=False)

            # Plot the correlations using a bar plot and annotate the bars with values
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x=correlation_df.index,
                             y=correlation_df['Correlation'], palette='coolwarm')
            plt.title(f'Correlations with {target_column}')
            plt.xticks(rotation=45)
            plt.xlabel('Columns')
            plt.ylabel('Correlation')

            # Annotate the bars with correlation values
            for i, v in enumerate(correlation_df['Correlation']):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

            plt.show()
