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

ridge_losses = {}
lasso_losses = {}

for lambda_val in lambda_vals:

    # instantiate ridge regression object
    ridge = MyRidgeRegression(lambda_val)

    # call to your CV function to compute mse for each fold
    losses = my_cross_val(ridge, "mse", X_train, y_train, k=10)

    # print mse from CV
    print("Ridge losses:", losses)
    print("Mean:", np.mean(losses))
    print("Std:", np.std(losses))
    print()

    ridge_losses[lambda_val] = np.mean(losses)

    # instantiate lasso object
    lasso = Lasso(lambda_val)

    # call to your CV function to compute mse for each fold
    losses = my_cross_val(lasso, "mse", X_train, y_train, k=10)

    # print mse from CV
    print("Lasso losses:", losses)
    print("Mean:", np.mean(losses))
    print("Std:", np.std(losses))
    print()

    lasso_losses[lambda_val] = np.mean(losses)

# instantiate ridge regression and lasso objects for best values of lambda
best_ridge_lambda = sorted(ridge_losses.keys(), key=lambda x : ridge_losses[x])[0]

best_lasso_lambda = sorted(lasso_losses.keys(), key=lambda x : lasso_losses[x])[0]

# fit models using all training data
ridge = MyRidgeRegression(best_ridge_lambda)
ridge.fit(X_train, y_train)

lasso = Lasso(best_lasso_lambda)
lasso.fit(X_train, y_train)

# predict on test data
ridge_predictions = ridge.predict(X_test)
lasso_predictions = lasso.predict(X_test)

# compute mse on test data
ridge_mse = (1 / y_test.shape[0]) * y_test.T @ ridge_predictions
lasso_mse = (1 / y_test.shape[0]) * y_test.T @ lasso_predictions

# print mse on test data
print("Ridge test mse:", ridge_mse)
print("Lasso test mse:", lasso_mse)

