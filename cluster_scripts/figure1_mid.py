import numpy as np 
from tqdm import tqdm 

from model import LogisticRegression
from removal_algorithm import removal

# read data 
data = np.load("data.npy", allow_pickle=True).item()
X_train = data['X_train']; X_test = data['X_test']; y_train = data['t_train']; y_test = data['t_test']

delta = 1e-4 
c = np.sqrt(2 * np.log(1.5 / delta))
gamma = 0.25 
m = 100 
num_remove = 100 

sigmas = np.logspace(-2, 2)[::-1]
lams = np.array([1e-4, 1e-3, 1e-2, 1e-1])
eps = np.logspace(-3, 4)
accuracies = np.zeros((len(lams), len(eps)))

for l_idx, l in enumerate(lams):
    for e_idx, e in tqdm(enumerate(eps)):
        # compute sigma 
        sigma = sigmas[e_idx]

        # fit model 
        regressor = LogisticRegression(learning_rate=0.001, num_iterations = 300, lam = l, sigma = sigma)
        regressor.fit(X_train, y_train)
        
        # remove the first 100 data points 
        weights, _ = removal(X_train, y_train, e, delta, sigma, l, gamma, num_remove, regressor.w, m, retrain = False)
        regressor.set_weights(weights)
        accuracies[l_idx, e_idx] = regressor.accuracy(X_test, y_test)

        print(f"accuracy[{l_idx}, {e_idx}] = {accuracies[l_idx, e_idx]}")
            
# save results 
np.savetxt("figure1_mid.txt", accuracies)
