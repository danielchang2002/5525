################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MySVM import MySVM

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

# change labels from 0 and 1 to -1 and 1 for SVM
y[y == 0] = -1

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# append column of 1s to include intercept
X = np.hstack((X, np.ones((num_data, 1))))
num_data, num_features = X.shape

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

#####################
# ADD YOUR CODE BELOW
#####################

MAX_ITERS = 10000
d = X_train.shape[1]
np.random.seed(42069)

# Import your CV package here (either your my_cross_val or sci-kit learn )
from my_cross_val import my_cross_val

eta_vals = [0.00001, 0.0001, 0.001]
C_vals = [0.01, 0.1, 1, 10, 100]

err_rates = {}

# SVM
for eta_val in eta_vals:
    for c_val in C_vals:

        # instantiate svm object
        svm = MySVM(d, MAX_ITERS, eta_val, c_val)

        # call to CV function to compute error rates for each fold
        scores = my_cross_val(svm, "err_rate", X_train, y_train, k=10)
    
        # print error rates from CV
        err_rates[(eta_val, c_val)] = scores

# instantiate svm object for best value of eta and C
columns = [f"F{i}" for i in range(1, 11)]

err_rate_df = pd.DataFrame.from_dict(err_rates, orient="index", columns=columns)
err_rate_df["Mean"] = err_rate_df.mean(axis=1)
err_rate_df["SD"] = err_rate_df.std(axis=1)
err_rate_df.index.name = "eta, c"
print(err_rate_df)
err_rate_df.to_csv("SVM.csv", float_format='%.5f')

best_hyperparam = err_rate_df.sort_values("Mean").index[0]
print("Best hyperparam:", best_hyperparam)
best_eta, best_c = best_hyperparam
svm = MySVM(d, MAX_ITERS, best_eta, best_c)

# fit model using all training data
svm.fit(X_train, y_train)

# predict on test data
pred = svm.predict(X_test)

# compute error rate on test data
err_rate = (1 / X_test.shape[0]) * (y_test != pred).sum()

# print error rate on test data
print("Test error rate:", err_rate)
