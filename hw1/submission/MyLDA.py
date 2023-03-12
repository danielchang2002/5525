import numpy as np

class MyLDA():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    def fit(self, X, y):

        # compute means
        X2 = X[y == 1]
        X1 = X[y == 0]
        m2 = X2.mean(axis=0)
        m1 = X1.mean(axis=0)

        # compute within class scatter
        Sw = np.zeros((X.shape[1], X.shape[1]))

        for x1i in X1:
            diff = np.expand_dims(x1i - m1, -1)
            result = diff @ diff.T
            Sw += result

        for x2i in X2:
            diff = np.expand_dims(x2i - m2, -1)
            result = diff @ diff.T
            Sw += result

        # compute w vector
        self.w = np.linalg.inv(Sw) @ (m2 - m1)

        # normalize
        self.w /= np.linalg.norm(self.w)
        self.w = np.expand_dims(self.w, -1)

    def predict(self, X):
        return 1 * ((X @ self.w) >= self.lambda_val).flatten()

