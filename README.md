# Complex-Valued Neural Networks (CVNNs): Exploring Activation Functions

## Overview
This repository contains code for an experiment aimed at evaluating the performance of different complex-valued activation functions within Complex-Valued Neural Networks (CVNNs). Using the MNIST dataset, the experiment investigates how activation functions like ModReLU, Complex Leaky ReLU, Complex tanh, Complex Cardioid, and zReLU behave in CVNNs.

## Experiment Design
The experiment utilizes the MNIST dataset and applies Fourier Transforms to the images, transforming them into the frequency domain. This aligns with the complex-valued nature of the activation functions being tested, allowing for an exploration of a diverse set of mathematical properties and behaviors within CVNNs. For a detailed description of the experiment design and activation functions, please refer to the Experiment Design section in the code comments.

## Metrics Used
* **Test Loss**: Measures the model's error on the test dataset.
* **Test Accuracy**: Evaluates how often the model's predictions match the true labels.
* **ROC AUC Score**: Represents the model's ability to discriminate between positive and negative classes.
* **Matthews Correlation Coefficient**: Measures the quality of binary classifications.
* **Cohen's Kappa**: Evaluates the agreement between the predicted and actual categories, taking into account the chance agreement.

## Requirements
* Python 3.x
* TensorFlow
* NumPy
* scikit-learn

## How to Run
1. Clone this repository.
2. Ensure that all dependencies are installed.
3. Run the main script:
   ```bash
   python main.py
