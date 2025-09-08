# 3 - MLP: Understanding Multi-Layer Perceptrons (MLPs)

This activity is designed to test our skills in Multi-Layer Perceptrons (MLPs).

## Exercise 1: Manual Calculation of MLP Steps

Consider a simple MLP with 2 input features, 1 hidden layer containing 2 neurons, and 1 output neuron. Use the hyperbolic tangent (tanh) function as the activation for both the hidden layer and the output layer. The loss function is mean squared error (MSE): L=1/N(y-^y)², where ^y is the networks output.

use the following specific values:

- Input and output vectors:
  - x = [0.5, -0.2]
  - y = 1.0

- Hidden layer weights:
  - W1 = [ [0.3, -0.1] , [0.2, 0.4] ]

- Hidden layer biases:
  - b1 = [0.1 , -0.2]

- Output layer weights:
  - W2 = [0.5 , -0.3]

- Output layer bias:
  - b2 = 0.2

- Learning rate:
  - n = 0.3

- Activation function:
  - tanh
  
Perform the following steps explicitly, showing all mathematical derivations and calculations with the provided values:

### Forward Pass

- Compute the hidden layer pre-activations
- Apply tanh to get hidden activations
- Compute the output pre-activation
- Compute the final output

### Loss Calculation

- Compute the MSE loss

### Backward Pass (Backpropagation)

Compute the gradients of the loss with respect to all weights and biases. Compute:

- Using tanh derivative
- Gradients for output layer
- Propagate to hidden layer
- Gradients for hidden layer

Show all intermediate steps and calculations.

### Parameter Update

Using the learning rate n = 0.1, update all weights and biases via gradient descent

Show all mathematical steps explicitly, including intermediate calculations (e.g., matrix multiplications, tanh applications, gradient derivations). Use exact numerical values throughout and avoid rounding excessively to maintain precision (at least 4 decimal places).

## Exercise 2: Binary Classification with Synthetic Data and Scratch MLP

Using the make_classification function from scikit-learn, generate a synthetic dataset with the following specifications:

- Number of Samples: 1000
- Number of Classes: 2
- Number of clusters per class: Use the n_clusters_per_class parameter creatively to achieve 1 cluster for one class and 2 for the other (hint: you may need to generate subsets separately and combine them, as the function applies the same number of clusters to all classes by default).
- Other parameters: Set n_features=2 for easy visualization, n_informative=2, n_redundant=0, random_state=42 for reproducibility, and adjust class_sep or flip_y as needed for a challenging but separable dataset.

Implement an MLP from scratch (without using libraries like TensorFlow or PyTorch for the model itself; you may use NumPy for array operations) to classify this data. You have full freedom to choose the architecture, including:

- Number of hidden layers (at least 1)
- Number of neurons per layer
- Activation functions (e.g., sigmoid, ReLU, tanh)
- Loss function (e.g., binary cross-entropy)
- Optimizer (e.g., gradient descent, with a chosen learning rate)

Steps to follow:

- Generate and split the data into training (80%) and testing (20%) sets.
- Implement the forward pass, loss computation, backward pass, and parameter updates in code.
- Train the model for a reasonable number of epochs (e.g., 100-500), tracking training loss.
- Evaluate on the test set: Report accuracy, and optionally plot decision boundaries or confusion matrix.
- Submit your code and results, including any visualizations.

## Exercise 3: Multi-Class Classification with Synthetic Data and Reusable MLP

Use make_classification to generate a synthetic dataset with:

- Number of samples: 1500
- Number of classes: 3
- Number of features: 4
- Number of clusters per class: Achieve 2 clusters for one class, 3 for another, and 4 for the last (again, you may need to generate subsets separately and combine them, as the function doesn't directly support varying clusters per class).
- Other parameters: n_features=4, n_informative=4, n_redundant=0, random_state=42.

Implement an MLP from scratch to classify this data. You may choose the architecture freely, but for an extra point (bringing this exercise to 4 points), reuse the exact same MLP implementation code from Exercise 2, modifying only hyperparameters (e.g., output layer size for 3 classes, loss function to categorical cross-entropy if needed) without changing the core structure.

Steps:

- Generate and split the data (80/20 train/test).
- Train the model, tracking loss.
- Evaluate on test set: Report accuracy, and optionally visualize (e.g., scatter plot of data with predicted labels).
- Submit code and results.

## Exercise 4: Multi-Class Classification with Deeper MLP

Repeat Exercise 3 exactly, but now ensure your MLP has at least 2 hidden layers. You may adjust the number of neurons per layer as needed for better performance. Reuse code from Exercise 3 where possible, but the focus is on demonstrating the deeper architecture. Submit updated code, training results, and test evaluation.
