################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MyLogisticRegression import MyLogisticRegression

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

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

MAX_ITERS = 1000
d = X_train.shape[1]

# Import your CV package here (either your my_cross_val or sci-kit learn )
from sklearn.metrics import cross_val_score

eta_vals = [0.000001, 0.00001, 0.0001, 0.001, 0.01]

# Logistic Regression
for eta_val in eta_vals:

    # instantiate logistic regression object
    logreg = MyLogisticRegression(d, MAX_ITERS, eta_val)

    # call to CV function to compute error rates for each fold
    logreg.fit(X_train, y_train)

    # print error rates from CV

# instantiate logistic regression object for best value of eta

# fit model using all training data

# predict on test data

# compute error rate on test data

# print error rate on test data
