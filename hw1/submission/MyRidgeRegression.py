import numpy as np

class MyRidgeRegression():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    def fit(self, X, y):
        self.weights = np.linalg.inv(X.T @ X + self.lambda_val * np.eye(X.shape[1])) @ X.T @ y

    def predict(self, X):
        return X @ self.weights

