import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from complex_valued_neural_networks.activation_functions import complex_cardioid, complex_leaky_relu, modrelu

import constants
from common import plot
from common.utils import MetricsHistory
from models.model_definition import create_model


def main():
    results = {}
    model_predictions = {}

    # Load MNIST dataset containing grayscale images of handwritten digits
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Apply Fourier Transform to the images, transforming them from spatial domain to frequency domain
    X_train = np.fft.fft2(X_train)
    X_test = np.fft.fft2(X_test)

    # Reshape the data to include a channel dimension, making it suitable for convolutional layers
    X_train = X_train.reshape((-1, 28, 28, 1))
    X_test = X_test.reshape((-1, 28, 28, 1))

    # Convert labels to one-hot encoding, making them suitable for categorical cross-entropy loss
    y_train = to_categorical(y_train, num_classes=constants.NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=constants.NUM_CLASSES)

    y_test_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoded true labels to integers

    # A dictionary containing various activation functions to be evaluated
    activation_functions = {
        'ReLU': 'relu',
        'ModReLU': modrelu,
        'Complex Leaky ReLU': complex_leaky_relu,
        'Complex Cardioid': complex_cardioid,
    }

    metrics_histories = {
        'Training Loss': {},
        'Validation Accuracy': {},
        'Matthews Correlation Coefficient': {},
        'Cohen’s Kappa Score': {},
        'ROC AUC Score': {}
    }

    # Iterate through activation functions, training a separate model for each
    for model_name, activation_function in activation_functions.items():
        print(f'Training model with {model_name} activation function.')
        # Initialize the validation data and the custom MetricsHistory callback
        validation_data = (X_test, y_test)
        test_data = (X_test, y_test)  # In this case, test data is same as validation data
        metrics_history = MetricsHistory(validation_data=validation_data, test_data=test_data)        
        # Train the model with the custom callback
        model = create_model(activation_function)
        history = model.fit(
            X_train, y_train, epochs=constants.NUM_EPOCHS,
            validation_data=validation_data,
            callbacks=[metrics_history]
        )
        
        # Store metrics history for plotting later
        metrics_histories['Training Loss'][model_name] = history.history['loss']
        metrics_histories['Validation Accuracy'][model_name] = history.history['val_accuracy']
        metrics_histories['Matthews Correlation Coefficient'][model_name] = metrics_history.mccs
        metrics_histories['Cohen’s Kappa Score'][model_name] = metrics_history.kappas
        metrics_histories['ROC AUC Score'][model_name] = metrics_history.auc_scores

        # Store predictions for later use
        y_pred = model.predict(X_test)
        model_predictions[model_name] = np.argmax(y_pred, axis=1)

        loss, accuracy = model.evaluate(X_test, y_test)
        y_pred_labels = np.argmax(y_pred, axis=1)

        mcc = matthews_corrcoef(y_test_labels, y_pred_labels)
        kappa = cohen_kappa_score(y_test_labels, y_pred_labels)
        auc_score = roc_auc_score(y_test, y_pred)

        results[model_name] = {
            'Test Loss': loss,
            'Test accuracy': accuracy,
            'ROC AUC Score': auc_score,
            'Matthews Correlation Coefficient': mcc,
            'Cohen\'s Kappa': kappa,
        }

    for model_name, y_pred_labels in model_predictions.items():
        plot.plot_confusion_matrix(y_test_labels, y_pred_labels, model_name)

    plot.plot_metrics_comparison(metrics_histories)

    # Print final results after all models are trained and plotted
    for model_name, result in results.items():
        print(f'\nFinal results for {model_name}:')
        for metric_name, metric_value in result.items():
            print(f'  {metric_name}: {metric_value}')

if __name__ == "__main__":
    main()
