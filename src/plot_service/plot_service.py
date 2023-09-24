import matplotlib as plt
import seaborn as sns


class PlotService():
    def __init__(self) -> None:
        pass

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
