import numpy as np 
from removal_algorithm import removal
from tqdm import tqdm 

from model import LogisticRegression
data = np.load("data.npy", allow_pickle=True).item()
X_train = data['X_train']; X_test = data['X_test']; y_train = data['t_train']; y_test = data['t_test']


# Data dependent bound for single and batch removal 
supported_removals = np.array([10] * 100)
single_removal_norm = np.zeros((len(supported_removals)))
batch_removal_norm = np.zeros((len(supported_removals)))


sigma = 0.1
lam = 1e-1 
eps = 1e1 
delta = 1e-4 
gamma = 0.25 

# fit model 
regressor = LogisticRegression(learning_rate=0.001, num_iterations = 300, lam = lam, sigma = sigma)
regressor.fit(X_train, y_train)
weights_single = regressor.w.copy()
weights_batch = regressor.w.copy() 

for i in tqdm(range(len(supported_removals))):
    print(f"round {i + 1} / {len(supported_removals)}")
    
    # unlearn points 
    weights_single, res_norm_single = removal(X_train[i * 10:], y_train[i * 10:], eps, delta, sigma, lam, gamma, 10, weights_single, m = 1)
    weights_batch, res_norm_batch = removal(X_train[i * 10:], y_train[i * 10:], eps, delta, sigma, lam, gamma, 10, weights_batch, m = 10)
    
    if i == 0:
        single_removal_norm[i] = res_norm_single 
        batch_removal_norm[i] = res_norm_batch 
    else: 
        single_removal_norm[i] = res_norm_single + single_removal_norm[i - 1]
        batch_removal_norm[i] = res_norm_batch + batch_removal_norm[i - 1]



removal_norms = np.c_[single_removal_norm, batch_removal_norm]
np.savetxt("figure2_data_norms.txt", removal_norms)


