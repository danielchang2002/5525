import numpy as np

def my_cross_val(model, loss_func, X, y, k=10):
  """
  Performs k-fold cross-validation

  Args:
    model (sklearn.linear_model): has fit() and predict() methods
    loss_func (str): either 'mse' or 'err_rate'
    X (np.array): shape = (n, d), n = # of data points, d = # of features  
    y (np.array): shape = (n, d)
    k (int): the number of folds

  Returns:
    losses (list): list of len k of the loss for each test fold
  """
  assert(loss_func in ["mse", "err_rate"])

  n, d = X.shape
  print(n)

  # precompute indices for fold partitions
  perm = np.random.permutation(n)

  losses = []

  for i in range(k):
    # get test fold
    test_start = i * (n // k)
    test_end = test_start + (n // k) if i != k - 1 else n
    test_indices = perm[test_start:test_end]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # get train fold
    train_indices = np.array(list(perm[:test_start]) + list(perm[test_end:]))
    X_train = X[train_indices]
    y_train = y[train_indices]

    # fit and get predictions
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # compute loss
    if loss_func == "mse":
      loss = (1 / n) * y_test.T @ predictions
    else:
      loss = (1 / n) * (y_test != predictions).sum()
    losses.append(loss)

  return losses


