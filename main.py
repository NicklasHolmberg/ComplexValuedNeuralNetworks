from activation_functions import complex_cardioid, complex_leaky_relu, complex_tanh, modrelu, zrelu
import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from model_definition import create_model
import constants

def main():
    results = {}

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

    # A dictionary containing various activation functions to be evaluated
    activation_functions = {
        'ReLU': 'relu',
        'ModReLU': modrelu,
        'Complex Leaky ReLU': complex_leaky_relu,
        'Complex tanh': complex_tanh,
        'Complex Cardioid': complex_cardioid,
        'zReLU': zrelu
    }

    # Iterate through activation functions, training a separate model for each
    for model_name, activation_function in activation_functions.items():
        # Create and train the model using the specified activation function
        model = create_model(activation_function)
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(X_test, y_test)

        # Predict class probabilities for additional metrics
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        # Compute various performance metrics
        mcc = matthews_corrcoef(y_test_labels, y_pred_labels)
        kappa = cohen_kappa_score(y_test_labels, y_pred_labels)
        auc_score = roc_auc_score(y_test, y_pred)

        # Store results
        results[model_name] = {
            'Test loss': loss,                      # Model's loss on the test data
            'Test accuracy': accuracy,              # Fraction of correctly classified samples
            'ROC AUC Score': auc_score,             # Receiver Operating Characteristic Area Under Curve
            'Matthews Correlation Coefficient': mcc,# Measure of quality for binary classification
            'Cohen\'s Kappa': kappa,                # Measure of classification accuracy normalized by chance
        }

    # Print results
    for model_name, result in results.items():
        print(f'{model_name}:')
        for metric_name, metric_value in result.items():
            print(f'  {metric_name}: {metric_value}')
        print()

if __name__ == "__main__":
    main()

