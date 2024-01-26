# Complex-Valued Neural Networks (CVNNs): Exploring Activation Functions

## Overview
This repository presents an experimental framework for evaluating various complex-valued activation functions when used in a Complex-Valued Neural Network (CVNN). It extends TensorFlow support for complex-valued data and includes specialized TensorFlow layers tailored for handling complex numbers. Primarily focusing on the MNIST dataset, the framework explores the performance of complex-valued activation functions like ModReLU, Complex Leaky ReLU, Complex tanh, Complex Cardioid, and zReLU. The aim is to understand how these functions influence the learning dynamics and performance of networks processing complex-valued data.

## Documentation
For a detailed background and a mathematical exploration of the principles behind these activation functions when used in a CVNN context, please refer to my exploratory paper published here in this repo: [Navigating_the_Complex_Plane.pdf](https://github.com/NicklasHolmberg/ComplexValuedNeuralNetworks/blob/main/Navigating_the_Complex_Plane.pdf)

## Experiment Design
There are numerous possible approaches for structuring a relevant experiment, and I encourage users to clone the repo and modify the main.py file to design their own experiment setups using different datasets and approaches. This particular experiment utilizes the MNIST dataset - a standard benchmark in image recognition comprising 70,000 labeled 28x28 grayscale images of handwritten digits - to examine the effects of complex-valued activation functions in a CVNN. The experiment includes a preprocessing step where the images are converted into the frequency domain via Fourier Transforms, effectively transforming the data into a complex-valued format. This transformation allows the neural networks to potentially leverage frequency patterns for image classification, which could offer insights into the effectiveness of complex-valued activation functions like ModReLU, Complex Leaky ReLU, and Complex Cardioid in a domain that differs from the traditional spatial analysis of image data. 

Key aspects of the implementation include:
- Custom TensorFlow Layers: ComplexDense and ComplexFlatten layers have been implemented to handle complex-valued data.
- Complex-Valued Activation Functions: These include ModReLU, Complex Leaky ReLU, Complex tanh, Complex Cardioid, zReLU, etc., all designed to operate on complex numbers.
- Custom Initializers: ComplexXavierInitializer and other initializers are used for effectively initializing the weights of complex-valued layers.

## Metrics Used
We evaluate model performance using a variety of metrics:
- Test Loss: Quantifies the prediction error on the test dataset.
- Test Accuracy: Measures the proportion of correct predictions.
- Confusion Matrix: Provides a detailed breakdown of the model's predictions, showing the number of correct and incorrect predictions for each class. This matrix helps in understanding the performance of the model for each individual class and identifying any biases or weaknesses in the classification.
- ROC AUC Score: Assesses the model's discriminative capability.
- Matthews Correlation Coefficient: Evaluates binary classification quality.
- Cohen's Kappa: Measures agreement between predicted and actual categories, accounting for chance agreement.

These metrics collectively provide a holistic view of the models' performance across different dimensions.

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
* Modularity and Extensibility: The code is structured to allow easy modification and extension. Developers can integrate new complex-valued layers, activation functions and datasets with minimal changes.