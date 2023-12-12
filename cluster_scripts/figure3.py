
import numpy as np
from tqdm import tqdm 

from removal_algorithm import loss_derivative, loss_derivative_2
from model import LogisticRegression


# read data
print("read data")
data = np.load("data.npy", allow_pickle=True).item()
X_train = data['X_train']; X_test = data['X_test']; y_train = data['t_train']; y_test = data['t_test']

lam = 1e-2
sigma = 1.0
regressor = LogisticRegression(lam = lam, sigma = sigma)
regressor.fit(X_train, y_train)

# init parameters 
removal_norm = np.zeros((X_train.shape[0], 1))
gamma = 0.25
eps = 10 
num_remove = 1
m = 1
delta = 1e-4 
X_train = np.insert(X_train, 0, 1, axis=1)
for i in tqdm(range(X_train.shape[0])):
    X_ = np.r_[X_train[:i], X_train[i+1:]]
    y_ = np.r_[y_train[:i], y_train[i+1:]]

    Delta = loss_derivative(X_train[i], y_train[i], regressor.w) + lam * regressor.w 
    H = loss_derivative_2(X_, y_, regressor.w, lam)
    H_delta = np.linalg.inv(H) @ Delta

    removal_norm[i] = np.linalg.norm(H_delta)

    if i % 100 == 0:
        np.savetxt(f"figure_3_{i}.txt", removal_norm)

np.savetxt("figure3.txt", removal_norm)