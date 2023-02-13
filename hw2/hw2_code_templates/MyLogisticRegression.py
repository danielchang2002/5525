import numpy as np

class MyLogisticRegression:

    def __init__(self, d, max_iters, eta):
        self.w = np.random.uniform(low=-0.01, high=0.01, size=d)
        self.epsilon = 10e-6 # pre-specified threshold for convergence detection
        self.max_iters = max_iters
        self.eta = eta
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True

        z = sigmoid(X @ self.w)
        print(z)

        for i in self.max_iters:
            pass

    def predict(self, X):
        if not self.fitted: 
            raise Exception("bruh you have to call fit before you call predict")
        return X @ self.w > 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))
    