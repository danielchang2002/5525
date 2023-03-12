################################
# DO NOT EDIT THE FOLLOWING CODE
################################
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
import numpy as np

from MyRidgeRegression import MyRidgeRegression
from my_cross_val import my_cross_val

# load dataset
X, y = fetch_california_housing(return_X_y=True)

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

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

lambda_vals = [0.01, 0.1, 1, 10, 100]

#####################
# ADD YOUR CODE BELOW
#####################

np.set_printoptions(precision=3)
import pandas as pd

ridge_losses = {}
lasso_losses = {}

for lambda_val in lambda_vals:
    # instantiate ridge regression object
    ridge = MyRidgeRegression(lambda_val)

    # call to your CV function to compute mse for each fold
    losses = my_cross_val(ridge, "mse", X_train, y_train, k=10)

    # save losses
    ridge_losses[lambda_val] = losses

    # instantiate lasso object
    lasso = Lasso(lambda_val)

    # call to your CV function to compute mse for each fold
    losses = my_cross_val(lasso, "mse", X_train, y_train, k=10)

    # save losses
    lasso_losses[lambda_val] = losses

columns = [f"F{i}" for i in range(1, 11)]

ridge_df = pd.DataFrame.from_dict(ridge_losses, orient="index", columns=columns)
ridge_df["Mean"] = ridge_df.mean(axis=1)
ridge_df["SD"] = ridge_df.std(axis=1)
ridge_df.index.name="lambda"

lasso_df = pd.DataFrame.from_dict(lasso_losses, orient="index", columns=columns)
lasso_df["Mean"] = lasso_df.mean(axis=1)
lasso_df["SD"] = lasso_df.std(axis=1)
lasso_df.index.name = "lambda"

print("Ridge k-fold losses:")
print(ridge_df)
ridge_df.to_csv("ridge_kfold.csv", float_format='%.3f')

print("\n\n\n")

print("Lasso k-fold losses:")
print(lasso_df)
lasso_df.to_csv("lasso_kfold.csv", float_format='%.3f')

print("\n\n\n")

# instantiate ridge regression and lasso objects for best values of lambda
best_ridge_lambda = ridge_df.sort_values("Mean").index[0]
best_lasso_lambda = lasso_df.sort_values("Mean").index[0]
print("Best ridge lambda:", best_ridge_lambda)
print("Best lasso lambda:", best_lasso_lambda)

# fit models using all training data
ridge = MyRidgeRegression(best_ridge_lambda)
ridge.fit(X_train, y_train)

lasso = Lasso(best_lasso_lambda)
lasso.fit(X_train, y_train)

# predict on test data
ridge_predictions = ridge.predict(X_test)
lasso_predictions = lasso.predict(X_test)

# compute mse on test data
ridge_mse = (1 / y_test.shape[0]) * (y_test.T - ridge_predictions) @ (y_test.T - ridge_predictions)
lasso_mse = (1 / y_test.shape[0]) * (y_test.T - lasso_predictions) @ (y_test.T - lasso_predictions)

# print mse on test data
print("Ridge test mse:", ridge_mse)
print("Lasso test mse:", lasso_mse)

