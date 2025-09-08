# Spotify Classification project

## Author - Arthur Carvalho

In this project, you will tackle a real-world classification task using a Multi-Layer Perceptron (MLP) neural network. The goal is to deepen your understanding of neural networks by handling data preparation, model implementation, training strategies, and evaluation without relying on high-level deep learning libraries. You will select a public dataset suitable for classification, process it, build and train your MLP, and analyze the results.

## Dataset Selection

Choose a public dataset for a classification problem. Sources include:

- Kaggle (e.g., datasets for digit recognition, spam detection, or medical diagnosis).
  - UCI Machine Learning Repository (e.g., Banknote Authentication, Adult Income, or Covertype).
  - Other open sources like OpenML, Google Dataset Search, or government data portals (e.g., data.gov).
  - Also, consider datasets from LOTS, here you have direct access to business problems.
- Ensure the dataset has at least 1,000 samples and multiple features (at least 5) to make the MLP meaningful.

- If selecting from a competition platform, note the competition rules and ensure your work complies.
- In your report: Provide the dataset name, source URL, size (rows/columns), and why you chose it (e.g., relevance to real-world problems, complexity).

## Dataset Explanation

- Describe the dataset in detail: What does it represent? What are the features (inputs) and their types (numerical, categorical)? What is the target variable (classes/labels)?
- Discuss any domain knowledge: E.g., if it's a medical dataset, explain key terms.
- Identify potential issues: Imbalanced classes, missing values, outliers, or noise.
- In your report: Include summary statistics (e.g., mean, std dev, class distribution) and visualizations (e.g., histograms, correlation matrices).

## Data Cleaning and Normalization

- Clean the data: Handle missing values (impute or remove), remove duplicates, detect and treat outliers.
- Preprocess: Encode categorical variables (e.g., one-hot encoding), normalize/scale numerical features (e.g., min-max scaling or z-score standardization).
- You may use libraries like Pandas for loading/cleaning and SciPy/NumPy for normalization.
- In your report: Explain each step, justify choices (e.g., "I used median imputation for missing values to avoid skew from outliers"), and show before/after examples (e.g., via tables or plots).

## MLP Implementation

- Code an MLP from scratch using only NumPy (or equivalent) for operations like matrix multiplication, activation functions, and gradients.
- Architecture: At minimum, include an input layer, one hidden layer, and output layer. Experiment with more layers/nodes for better performance.
- Activation functions: Use sigmoid, ReLU, or tanh.
- Loss function: Cross-entropy for classification.
- Optimizer: Stochastic Gradient Descent (SGD) or a variant like mini-batch GD.
- Pre-built neural network libraries allowed, but you must understand and explain all parts of the code and analysis submitted.
- In your report: Provide code or key code snippets (the full code). Explain hyperparameters (e.g., learning rate, number of epochs, hidden units).

## Model Training

- Train your MLP on the prepared data.
- Implement the training loop: Forward propagation, loss calculation, backpropagation, and parameter updates.
- Handle initialization (e.g., random weights) and regularization if needed (e.g., L2 penalty, but optional).
- In your report: Describe the training process, including any challenges (e.g., vanishing gradients) and how you addressed them.

## Training and Testing Strategy

- Split the data: Use train/validation/test sets (e.g., 70/15/15 split) or k-fold cross-validation.
- Training mode: Choose batch, mini-batch, or online (stochastic) training; explain why (e.g., "Mini-batch for balance between speed and stability").
- Early stopping or other techniques to prevent overfitting.
- In your report: Detail the split ratios, random seeds for reproducibility, and rationale. Discuss validation's role in hyperparameter tuning.

## Error Curves and Visualization

- Plot training and validation loss/accuracy curves over epochs.
- Use Matplotlib or similar for plots.
- Analyze: Discuss convergence, overfitting/underfitting, and adjustments made.
- In your report: Include at least two plots (e.g., loss vs. epochs, accuracy vs. epochs). Interpret trends (e.g., "Loss plateaus after 50 epochs, indicating convergence").

## Evaluation Metrics

- Apply classification metrics on the test set: Accuracy, precision, recall, F1-score, confusion matrix (for multi-class).
- If imbalanced, include ROC-AUC or precision-recall curves.
- Compare to baselines (e.g., majority class predictor).
- In your report: Present results in tables (e.g., metric values) and visualizations (e.g., confusion matrix heatmap). Discuss strengths/weaknesses (e.g., "High recall on class A but low on B due to imbalance").
