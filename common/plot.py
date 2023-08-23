import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.signal import savgol_filter

import constants

# Apply Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")

def plot_metrics_comparison(metrics_histories):
    """
    Plots the comparison of various metrics across different models.

    Args:
    - metrics_histories (dict): A dictionary containing the history of metrics for each model.
    - num_epochs (int): The number of epochs the models were trained for.
    """
    epochs = range(1, constants.NUM_EPOCHS + 1)
    sns.set_palette("husl")  # Sets a color palette

    for metric_name, metric_values in metrics_histories.items():
        plt.figure(figsize=(10, 6))
        for model_name, values in metric_values.items():
            # Optionally apply smoothing if appropriate
            values_smoothed = savgol_filter(values, 11, 3) if len(values) > 10 else values
            plt.plot(epochs, values_smoothed, label=model_name, lw=2)  # Smoothed, thicker line
        plt.title(f'Comparison of {metric_name} Across Activation Functions')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """
    Plots the confusion matrix for the given true labels and predictions.
    
    Args:
    - y_true (numpy.array): Array of true labels.
    - y_pred (numpy.array): Array of predicted labels.
    - model_name (str): Name of the model to include in the plot title.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()