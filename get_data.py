from sklearn.datasets import fetch_openml
import numpy as np 

def get_data():
    # get mnist dataset 
    mnist = fetch_openml('mnist_784', version = 1)
    images, labels = mnist['data'].to_numpy(), mnist['target'].to_numpy()

    # extract the 3's and 8's
    digits = ['3', '8']
    idx = np.logical_or(labels == digits[0], labels == digits[1])

    # extract digits of interest
    X = images[idx, :]
    t = labels[idx]
    t[t == digits[0]] = 0; t[t == digits[1]] = 1
    t = t.astype(float)

    X_train = X[:11000, :];  X_test = X[11000:, :]
    t_train = t[:11000];  t_test = t[11000:]

    # standardize training set and test set
    Xm, Xs = X_train.mean(0), X_train.std(0)
    Xs[Xs == 0] = 1 # avoid division by zero for "always black" pixels

    X_train_std = (X_train - Xm)/Xs
    X_test_std = (X_test - Xm)/Xs

    return X_train_std, t_train, X_test_std, t_test