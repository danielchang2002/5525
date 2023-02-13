import numpy as np

class MyLogisticRegression:

    def __init__(self, d, max_iters, eta):
        self.w = np.random.uniform(low=-0.01, high=0.01, size=d)
        self.epsilon = 10e-6 # pre-specified threshold for convergence detection
        self.max_iters = max_iters
        self.eta = eta
        self.fitted = False

    def fit(self, X, y):
        n = X.shape[0]
        self.fitted = True

        loss = np.inf
        converged = False
        num_iters_elapsed = -1

        for i in range(self.max_iters):
            num_iters_elapsed = i
            z = sigmoid(X @ self.w)
            gradient = -(1 / n) * (y - z) @ X
            self.w -= self.eta * gradient
            this_loss = self.get_loss(X, y)
            if loss - this_loss < self.epsilon:
                converged = True
                break
            loss = this_loss

        # if converged:
        #     print(f"Converged after {num_iters_elapsed} iterations")
        # else:
        #     print("bruh you did not converge")
        # print(f"Loss: {loss}")

    def get_loss(self, X, y):
        return - np.mean(y * np.log(sigmoid(X @ self.w)) + (1 - y) * np.log(1 - sigmoid(X @ self.w)))

    def predict(self, X):
        if not self.fitted: 
            raise Exception("bruh you have to call fit before you call predict")
        return X @ self.w > 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))
    