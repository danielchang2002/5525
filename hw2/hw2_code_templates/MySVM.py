import numpy as np

class MySVM:

    def __init__(self, d, max_iters, eta, c):
        self.w = np.random.uniform(low=-0.01, high=0.01, size=d)
        self.epsilon = 10e-6 # pre-specified threshold for convergence detection
        self.max_iters = max_iters
        self.eta = eta
        self.c = c
        self.fitted = False

    def fit(self, X, y):
        n = X.shape[0]
        self.fitted = True

        loss = np.inf
        converged = False
        num_iters_elapsed = -1

        for i in range(self.max_iters):

            # compute gradient
            keep_index = 1 - y * (X @ self.w) > 0
            gradient = self.c * -y[keep_index] @ X[keep_index] + self.w

            # update
            self.w -= self.eta * gradient

            # check for convergence
            num_iters_elapsed = i
            this_loss = self.get_loss(X, y)
            if loss - this_loss < self.epsilon:
                converged = True
                break
            loss = this_loss

    def get_loss(self, X, y):
        dot_prods = 1 - y * (X @ self.w)
        dot_prods[dot_prods < 0] = 0
        return 0.5 * (self.w @ self.w) + self.c * np.sum(dot_prods)


    def predict(self, X):
        if not self.fitted: 
            raise Exception("bruh you have to call fit before you call predict")
        return -1 + 2 * (X @ self.w > 0)
