# define functions 
import numpy as np 
from model import LogisticRegression
from copy import deepcopy
from matplotlib import pyplot as plt 
import pickle 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import seaborn as snb 
snb.set_theme()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def loss_derivative(X_batch, y_batch, weights):
    preds = sigmoid(np.dot(X_batch, weights))
    return np.dot(X_batch.T, (preds - y_batch)) # / len(y_batch) 

def loss_derivative_2(X_batch, y_batch, weights, lam):
    preds = sigmoid(np.dot(X_batch, weights))
    S = np.diag(preds * (1 - preds))
    return X_batch.T @ S @ X_batch + len(y_batch) * lam * np.eye(X_batch.shape[1])

def spectral_norm(A, num_iters=20):
    x = np.random.randn(A.shape[0])
    norm = 1
    for _ in range(num_iters):
        x = A @ x
        norm = np.sqrt(x.T @ x)
        x /= norm
    return np.sqrt(norm)

def removal(X, y, eps, delta, sigma, lam, gamma, num_remove, w, m, retrain = True, bias = True, plot_trace = False):
    weights = deepcopy(w)

    # # # init variables # # # 
    beta = 0
    beta_acc = 0 
    B = np.arange(0, num_remove) % m  
    c = np.sqrt(2 * np.log(1.5 / delta))
    if bias:
        X = np.insert(X, 0, 1, axis=1)
    K = X.T @ X
    retrained = False 

    # # Loop # # 
    for j in range(num_remove // m):
        Bj = B[j * m:(j+1) * m]
        Delta = loss_derivative(X[Bj], y[Bj], weights) + len(Bj) * lam * weights
        H = loss_derivative_2(X[Bj[-1] + 1:], y[Bj[-1] + 1:], weights, lam)
        for i in Bj:
            K -= X[i][:, None] @ X[i][None, :]
        X = np.delete(X, Bj, axis = 0)
        y = np.delete(y, Bj)

        H_delta = np.linalg.inv(H) @ Delta
        norm = gamma * spectral_norm(K) * np.linalg.norm(H_delta) * np.linalg.norm(X[Bj] @ H_delta)
        beta += norm 
        beta_acc += norm 

        # check budget 
        if beta > sigma * eps / c:
            print(f"Retraining. beta = {beta:2.2f}, budget = {sigma * eps / c:2.2f}")
            retrained = True    
            if not retrain:
                if num_remove == m:
                    return weights + H_delta.squeeze(), beta_acc 
                else:
                    weights += H_delta.squeeze()

            else: 
                # retrain on remaining data points 
                regressor = LogisticRegression(learning_rate=0.001, num_iterations = 300, lam = lam, sigma = sigma, add_bias=bias)
                if bias: 
                    regressor.fit(X[:, 1:], y)
                else: 
                    regressor.fit(X, y)
                weights = regressor.w       
                beta = 0  
                
        else:
            weights += H_delta.squeeze()

    if plot_trace:
        return weights, retrained  
    else:
        return weights, beta_acc 


def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    return cb 

def plot_weight_trace(X, y, eps, delta, sigma, lam, gamma, regressor, num_frames):
    w = regressor.w 
    x_grid = np.linspace(-4.5, 4.5)
    decision = -w[0]*x_grid/w[1]

    limits = [w[0] - 2, w[0] + 2, w[1] - 2, w[1] + 2]
    param1 = np.linspace(limits[0], limits[1], 300)
    param2 = np.linspace(limits[2], limits[3], 300)

    retrained_flag = False 

    fig, ax = plt.subplots(1, 2, figsize = (12, 6))
    def update(frame):
        nonlocal w, decision, retrained_flag
        # clear plot     
        ax[0].cla(); ax[1].cla()

        # update parameters 
        _, _, losses, optimal_param1, optimal_param2 = regressor.plot_parameter_contours(X[frame:], y[frame:], limits)

        # plot landscape 
        im = ax[0].contourf(param1, param2, losses, cmap="plasma")
        ax[0].scatter(optimal_param1, optimal_param2, marker='x', color='red', label='Optimum')
        if retrained_flag and False:
            ax[0].text(limits[0] + 1, limits[3] - 1, "RETRAINED!", color = [1, 0, 1], fontsize = 40)
        ax[0].set_xlabel('$\phi_1$')
        ax[0].set_ylabel('$\phi_2$')
        ax[0].set_title("Loss Landscape")
        
        
        cb = add_colorbar(im, fig, ax[0])
        ax[0].set_xlim([limits[0], limits[1]])
        ax[0].set_ylim([limits[2], limits[3]])


        # plot decision boundary 
        ax[1].plot(X[y==0, 0], X[y==0, 1], 'b.', label = 'class 1')
        ax[1].plot(X[y==1, 0], X[y==1, 1], 'r.', label = 'class 2')
        ax[1].plot(x_grid, decision, 'k-')
        ax[1].plot(X[frame, 0], X[frame, 1], 'o', color=[1, 0, 1], label = "Point to remove", alpha = 0.7)
        ax[1].plot(X[:frame, 0], X[:frame, 1], 'g.', label = "Removed points")
        ax[1].set_xlim([-3, 4.5])
        ax[1].set_ylim([-3, 4.5])

        # set legend 
        # ax[0].legend(loc = 'upper right')
        # ax[1].legend(loc='upper right')

        # forget next point 
        _ , retrained_flag = removal(X, y, eps, delta, sigma, lam, gamma, frame, w, frame, bias = False, plot_trace = True, retrain = False)
        regressor.set_weights(np.array([optimal_param1, optimal_param2]))
        decision = -optimal_param1*x_grid/optimal_param2

        # remove colorbar 
        cb.remove()

    ani = FuncAnimation(fig, update, frames = range(1, num_frames), repeat = False)
    ani.save("Results/parameter_trace.gif", writer = "imagemagick", fps = 3)
    plt.show()        