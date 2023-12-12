import numpy as np 
from tqdm import tqdm 

from model import LogisticRegression
from removal_algorithm import removal

# read data 
data = np.load("data.npy", allow_pickle=True).item()
X_train = data['X_train']; X_test = data['X_test']; y_train = data['t_train']; y_test = data['t_test']


eps = 1
gamma = 0.25 
delta = 1e-4 
sigmas = np.array([0.01, 0.1, 1, 10, 100])
lams = np.array([1e-4, 1e-3, 1e-2, 1e-1])
supported_removals = np.array([
        [1, 20, 50, 80, 100],
        [1, 250, 500, 750, 1000],
        [1, 750, 1500, 2250, 3000],
        [50, 2500, 5000, 7500, 10000]
])

accuracies = np.zeros((len(lams), 5))


for l_idx, lam in enumerate(lams): 
    for i in tqdm(range(5)):
        # fit model 
        regressor = LogisticRegression(learning_rate=0.001, num_iterations = 300, lam = lam, sigma = sigmas[i])
        regressor.fit(X_train, y_train)
        weights = regressor.w 
        
        # remove data points 
        m = 10 if supported_removals[l_idx, i] < 11 else supported_removals[l_idx, i] // 10 
        w_tilde, _ = removal(X_train, y_train, eps, delta, sigmas[i], lam, gamma, supported_removals[l_idx, i], weights, m, retrain = False)
        regressor.set_weights(w_tilde)
        accuracies[l_idx, i] = regressor.accuracy(X_test, y_test)

# save results 
np.savetxt("figure1_right.txt", accuracies)          
