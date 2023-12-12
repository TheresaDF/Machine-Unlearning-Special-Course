import numpy as np
from model import LogisticRegression
from tqdm import tqdm 

# read data 
data = np.load("data.npy", allow_pickle=True).item()
X_train = data['X_train']; X_test = data['X_test']; y_train = data['t_train']; y_test = data['t_test']


sigmas = np.logspace(-2, 2)
lams = np.array([1e-4, 1e-3, 1e-2, 1e-1])
accuracies = np.zeros((len(sigmas), len(lams)))
iter = 10 

for count in range(iter):
    for i in tqdm(range(len(sigmas))):
        for j in range(len(lams)):
            regressor = LogisticRegression(lam = lams[j], sigma = sigmas[i], seed = i)
            regressor.fit(X_train, y_train)
            accuracies[i, j] += regressor.accuracy(X_test, y_test)

accuracies = accuracies / iter 
# save results 
np.savetxt("figure1_left.txt", accuracies)