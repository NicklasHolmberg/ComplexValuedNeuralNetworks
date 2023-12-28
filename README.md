# Complex-Valued Neural Networks (CVNNs): Exploring Activation Functions

## Overview
This repository presents an experimental framework for evaluating various complex-valued activation functions in Complex-Valued Neural Networks (CVNNs). Focusing on the MNIST dataset, it explores the performance of activation functions like ModReLU, Complex Leaky ReLU, Complex tanh, Complex Cardioid, and zReLU within CVNNs. The aim is to understand how these functions affect the learning dynamics and performance of networks processing complex-valued data.

## Experiment Design
The experiment employs the MNIST dataset, where images are transformed into the frequency domain via Fourier Transforms. This transformation is crucial for aligning the data with the complex-valued nature of the networks. The CVNNs are built using TensorFlow and Keras, allowing for the integration of complex-valued operations within the familiar and powerful framework of these libraries.

Key aspects of the implementation include:
- Custom TensorFlow Layers: ComplexDense and ComplexFlatten layers have been implemented to handle complex-valued data.
- Complex-Valued Activation Functions: These include ModReLU, Complex Leaky ReLU, Complex tanh, Complex Cardioid, and zReLU, all designed to operate on complex numbers.
- Custom Initializers: ComplexXavierInitializer and other initializers are used for effectively initializing the weights of complex-valued layers.

## Metrics Used
We evaluate model performance using a variety of metrics:
- Test Loss: Quantifies the prediction error on the test dataset.
- Test Accuracy: Measures the proportion of correct predictions.
- ROC AUC Score: Assesses the model's discriminative capability.
- Matthews Correlation Coefficient: Evaluates binary classification quality.
- Cohen's Kappa: Measures agreement between predicted and actual categories, accounting for chance agreement.

These metrics collectively provide a holistic view of the models' performance across different dimensions.

## Requirements
- Python 3.x: The primary programming language used.
- TensorFlow: The core framework for building and training neural network models.
- Keras: Integrated within TensorFlow, used for its high-level neural networks API which simplifies model construction and experimentation.
- NumPy: Essential for numerical computations, particularly for data preprocessing.
- scikit-learn: Used for additional data processing and evaluation metrics.

## How to Run

1. **Clone this repository.**
   - Use `git clone https://github.com/NicklasHolmberg/ComplexValuedNeuralNetworks` to clone the repository to your local machine.

2. **Set up a Python virtual environment.**
   - Navigate to the cloned repository's directory.
   - Create a virtual environment: 
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On Unix or MacOS:
       ```bash
       source venv/bin/activate
       ```

3. **Install dependencies.**
   - With the virtual environment activated, install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the main script.**
   - Execute the main Python script:
     ```bash
     python main.py
     ```

## Additional Notes
* TensorFlow and Keras Integration: The custom complex-valued layers and activation functions are seamlessly integrated into TensorFlow's computational graph, ensuring efficient computation and gradient propagation.
* Modularity and Extensibility: The code is structured to allow easy modification and extension. Researchers and developers can integrate new complex-valued layers or activation functions with minimal changes.
* Experiment Customization: Users can easily modify the script to experiment with different network architectures, activation functions, or datasets.
This framework opens up avenues for exploring the potential of CVNNs in various applications where complex-valued data is prevalent, such as signal processing and communications.
