################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_cross_val import my_cross_val
from MyLDA import MyLDA

# load dataset
data = pd.read_csv('hw1_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

num_data, num_features = X.shape

plt.scatter(X[:1000, 0], X[:1000, 1])
plt.scatter(X[1000:, 0], X[1000:, 1])
plt.show()

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

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

# visualize the projected data points to see what lambda values make sense
lda = MyLDA(0)
lda.fit(X_train, y_train)
proj = X_train @ lda.w
plt.hist(proj[y_train == 1], bins=50, alpha=0.5)
plt.hist(proj[y_train == 0], bins=50, alpha=0.5)
plt.show()

lambda_vals = np.arange(-2, 2.1, 0.2)

lda_losses = {}

for lambda_val in lambda_vals:

    # instantiate LDA object
    lda = MyLDA(lambda_val)

    # call to your CV function to compute error rates for each fold
    losses = my_cross_val(lda, "err_rate", X_train, y_train, k=10)

    # print error rates from CV
    print(f"lambda = {lambda_val} losses:", losses)
    print("Mean:", np.mean(losses))
    print("Std:", np.std(losses))
    print()
    lda_losses[lambda_val] = np.mean(losses)

# instantiate LDA object for best value of lambda
best_lda_lambda = sorted(lda_losses.keys(), key=lambda x : lda_losses[x])[0]
print("best lambda:", best_lda_lambda)
lda = MyLDA(best_lda_lambda)

# fit model using all training data
lda.fit(X_train, y_train)

# predict on test data
pred = lda.predict(X_test)

# compute error rate on test data
err_rate = (1 / X_test.shape[0]) * (y_test != pred).sum()

# print error rate on test data
print("error rate:", err_rate)

