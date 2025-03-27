import math
import sys
import numpy as np
import csv

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

# Base class with shared functions and attributes
class EssayScorePredictorBase:
    """
    Base class containing shared feature extraction methods and data processing
    """

    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.training_loss = None
        self.validation_loss = None

    @staticmethod
    def calculate_length_4th_root(essay):
        """
        Calculates the 4th root of the number of words in the essay
        
        Returns:
            float: The 4th root of the word count
        """
        length = len(essay.split())
        return length ** 0.25

    @staticmethod
    def calculate_ovix(essay):
        """
        Calculates the OVIX value of the essay.
        OVIX = ln(n) / (ln(2) - (ln(k) / ln(n)))
        where n is total words and k is the number of unique words

        Returns:
            float: The computed OVIX value
        """
        words = essay.split()
        n = len(words)
        k = len(set(words))
        if n == 0 or k == 0 or n == 1:  # Handle case with only one word
            return 0.0
        return math.log(n) / (math.log(2) - (math.log(k) / math.log(n)))


    @classmethod
    def prepare_data(cls, data):
        """
        Processes a list of (essay, score) tuples into a feature matrix X and standardized scores y
        The standardization is performed as s'_i = (s_i - mean(s)) / std(s)
        
        Returns:
            tuple: A tuple containing:
                - X (np.array): Feature matrix with shape (n_samples, 2), 
                    each vector is [4th-root of length, OVIX]
                - y_standardized (np.array): Standardized scores
        """
        X = []
        y = []
        for essay, score in data:
            x1 = cls.calculate_length_4th_root(essay)
            x2 = cls.calculate_ovix(essay)
            X.append([x1, x2])
            y.append(score)
        X = np.array(X)
        y = np.array(y)
        
        # Compute the mean and standard deviation within the essay set
        mean_y = np.mean(y)
        std_y = np.std(y)
        # Guard against division by zero if std is 0
        if std_y == 0:
            std_y = 1e-8
            
        # Standardize the scores using the formula above
        y_standardized = (y - mean_y) / std_y
        return X, y_standardized

# Scikit-learn Based Predictor
class EssayScorePredictorSKLearn(EssayScorePredictorBase):
    """
    Uses scikit-learn's LinearRegression to predict essay scores
    """
    def __init__(self, learning_rate, epochs):
        super().__init__(learning_rate, epochs)
        self.model = LinearRegression()

    def train(self, training_data, validation_data):
        X_train, y_train = self.prepare_data(training_data)
        X_val, y_val = self.prepare_data(validation_data)
        
        for epoch in range(self.epochs):
            # Fit the model on the full training set
            self.model.fit(X_train, y_train)
            
            # Get predictions and compute mean squared errors
            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)
            self.training_loss = mean_squared_error(y_train, y_train_pred)
            self.validation_loss = mean_squared_error(y_val, y_val_pred)
            
            # Retrieve weights and bias (intercept)
            weights = self.model.coef_
            intercept = self.model.intercept_
            
            # print(f"Epoch {epoch+1}/{self.epochs}")
            # print(f"Weights: {weights}")
            # print(f"Intercept: {intercept}")
            # print(f"Training Loss: {self.training_loss}")
            # print(f"Validation Loss: {self.validation_loss}")
            # print("-" * 50)

    def predict(self, essay):
        """
        Predict the essay score for a given essay
        Returns:
            float: The predicted essay score
        """
        x1 = self.calculate_length_4th_root(essay)
        x2 = self.calculate_ovix(essay)
        return self.model.predict([[x1, x2]])[0]

# PyTorch Based Predictor
class LinearRegressionTorch(nn.Module):
    """
    A simple one-layer linear network for regression
    """
    def __init__(self):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        # Output tensor of shape (batch_size, 1)
        return self.linear(x)

class EssayScorePredictorTorch(EssayScorePredictorBase):
    """
    Uses PyTorch to build and train a neural network for essay score prediction
    """
    def __init__(self, learning_rate, epochs):
        super().__init__(learning_rate, epochs)
        self.model = LinearRegressionTorch()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, training_data, validation_data):
        X_train, y_train = self.prepare_data(training_data)
        X_val, y_val = self.prepare_data(validation_data)
        # Convert data to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            # Clipping to avoid exploding gradients (which lead to values to be nan)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Evaluate on validation set
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)

            self.training_loss = loss.item()
            self.validation_loss = val_loss.item()
            weights = self.model.linear.weight.data.numpy()
            bias = self.model.linear.bias.data.numpy()

            # if (epoch + 1) % 200 == 0 or epoch == 0 or (epoch + 1) == self.epochs:
            #     print(f"Epoch {epoch+1}/{self.epochs}")
            #     print(f"Weights: {weights}")
            #     print(f"Bias: {bias}")
            #     print(f"Training Loss: {self.training_loss}")
            #     print(f"Validation Loss: {self.validation_loss}")
            #     print("-" * 50)

    def predict(self, essay):
        """
        Predict the essay score for a given essay using the PyTorch model
        
        Returns:
            float: The predicted essay score
        """
        self.model.eval()
        x1 = self.calculate_length_4th_root(essay)
        x2 = self.calculate_ovix(essay)
        features = torch.tensor([[x1, x2]], dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(features)
        return pred.item()

# Main function to run both predictors
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 lab3_final.py /home/dsv/robe/lis050/lab3/data/asap-train.tsv")
        sys.exit(1)

    file_path = sys.argv[1]

    # Read the TSV data (columns "essay" and "domain1_score")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            essay = row['essay']
            score = float(row['domain1_score'])
            data.append((essay, score))

    # Split the data into two halves: training and validation
    split_index = len(data) // 2
    training_data = data[:split_index]
    validation_data = data[split_index:]

    print("Scikit-learn Predictor: ")
    predictor_sklearn = EssayScorePredictorSKLearn(learning_rate=0.1, epochs=1000)
    predictor_sklearn.train(training_data, validation_data)
    #First predictor (4th root of length) and the second predictor (OVIX) and bias (intercept)
    print("Final Weights (scikit-learn):", predictor_sklearn.model.coef_, predictor_sklearn.model.intercept_)
    print("Final Training Loss (scikit-learn):", predictor_sklearn.training_loss)
    print("Final Validation Loss (scikit-learn):", predictor_sklearn.validation_loss)
    print("Learning Rate (scikit-learn):", predictor_sklearn.learning_rate)
    print("Epochs (scikit-learn):", predictor_sklearn.epochs)

    print("\nPyTorch Predictor: ")
    predictor_torch = EssayScorePredictorTorch(learning_rate=0.1, epochs=1000)
    predictor_torch.train(training_data, validation_data)
    # Retrieve weights and bias from PyTorch model
    torch_weights = predictor_torch.model.linear.weight.data.numpy()
    torch_bias = predictor_torch.model.linear.bias.data.numpy()
    print("Final Weights (PyTorch):", torch_weights, torch_bias)
    print("Final Training Loss (PyTorch):", predictor_torch.training_loss)
    print("Final Validation Loss (PyTorch):", predictor_torch.validation_loss)
    print("Learning Rate (PyTorch):", predictor_torch.learning_rate)
    print("Epochs (PyTorch):", predictor_torch.epochs)