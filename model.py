import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle 

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

class LogisticRegression:
    def __init__(self, add_bias = True, learning_rate=0.01, num_iterations=1000, lam = 0.01, sigma = 0.01, batch_size = 512, seed = 0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.lam = lam
        self.sigma = sigma 
        self.batch_size = batch_size
        self.add_bias = add_bias 
        self.seed = seed 

        self.__set_seed()

    def __set_seed(self): 
        np.random.seed(seed = self.seed)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Add bias 
        if self.add_bias:
            X = np.insert(X, 0, 1, axis=1)

        # Initialize weights
        self.w = np.random.randn(X.shape[1])
        b = np.random.normal(0, self.sigma, X.shape[1])

        # optimization loop 
        for _ in range(self.num_iterations):
            # randomize order 
            indices = np.random.permutation(len(y))
            X_permuted = X[indices]
            y_permuted = y[indices]

            for i in range(0, len(y) - 1, self.batch_size):
                X_batch = X_permuted[i:i + self.batch_size]
                y_batch = y_permuted[i:i + self.batch_size]

                # compute predictions 
                preds = self.__sigmoid(np.dot(X_batch, self.w))

                # Compute the gradient of the cost function
                gradient = np.dot(X_batch.T, (preds - y_batch))  + len(y_batch) * self.lam * self.w + b 

                # Update weights 
                self.w -= self.learning_rate * gradient

    def predict(self, X):
        if self.add_bias:
            X = np.insert(X, 0, 1, axis=1)
        z = np.dot(X, self.w)
        h = self.__sigmoid(z)
        predictions = (h >  0.5).astype(int)
        return predictions

    def set_weights(self, weights):
        self.w = weights

    def accuracy(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def plot_parameter_contours(self, X, y, limits, plot = False):
        # only use this function without bias 

        # Generate parameter values for the contour plot
        num = 300
        param_range1 = np.linspace(limits[0], limits[1], num)
        param_range2 = np.linspace(limits[2], limits[3], num)
        param1, param2 = np.meshgrid(param_range1, param_range2)
        params = np.stack([param1.ravel(), param2.ravel()], axis = 0)

        # compute losses 
        predictions = self.__sigmoid(X @ params)
        losses = np.sum(np.multiply(-y[:, None], np.log(predictions)) - np.multiply((1 - y[:, None]), np.log(1 - predictions)), axis = 0) 
        losses += 0.5 * X.shape[0] * self.lam * np.sum(params ** 2, axis = 0) + [np.dot(np.random.normal(0, self.sigma, 2), params[:, i]) for i in range(params.shape[1])]
        losses = losses.reshape(num, num)

        # find optimal weights 
        optimal_indices = np.unravel_index(np.argmin(losses), losses.shape)
        optimal_param1 = param1[optimal_indices]
        optimal_param2 = param2[optimal_indices]

        # Plot the contour plot
        if plot: 
            fig, ax = plt.subplots(1, 1, figsize = (8, 6))
            im = ax.contourf(param1, param2, losses, cmap="RdBu")
            ax.scatter(optimal_param1, optimal_param2, marker='x', color='red', label='Optimal Parameters')
            ax.scatter(self.w[0], self.w[1], marker='x', color='blue', label='Optimal Parameters')
            ax.set_xlabel('$\phi_1$')
            ax.set_ylabel('$\phi_2$')
            ax.set_title('Contour Plot of Logistic Regression Parameters')
            ax.legend()
            add_colorbar(im, fig, ax)
            plt.show()
            
        return param1, param2, losses, optimal_param1, optimal_param2 
    

    # def compute_contours(self, X, y, indices, filename):

    #     # Generate parameter values for the contour plot
    #     num = 300
    #     param_range1 = np.linspace(-5, 1, num)
    #     param_range2 = np.linspace(-1, 5, num)
    #     param1, param2 = np.meshgrid(param_range1, param_range2)
    #     params = np.stack([param1.ravel(), param2.ravel()], axis = 0)

    #     # allocate space
    #     losses = np.zeros((num, num, len(indices)))
    #     phis_1 = np.zeros((len(indices)))
    #     phis_2 = np.zeros((len(indices)))
        
    #     # loop
    #     for j in indices: 
    #         # compute losses 
    #         predictions = self.__sigmoid(X[j:] @ params)
    #         loss = np.sum(np.multiply(-y[j:, None], np.log(predictions)) - np.multiply((1 - y[j:, None]), np.log(1 - predictions)), axis = 0) 
    #         loss += 0.5 * X[:j].shape[0] * self.lam * np.sum(params ** 2, axis = 0) + [np.dot(np.random.normal(0, self.sigma, 2), params[:, i]) for i in range(params.shape[1])]
    #         loss = loss.reshape(num, num)

    #         print(f"shape og loss = {loss.shape}")

    #         # find optimal weights 
    #         optimal_indices = np.unravel_index(np.argmin(loss), loss.shape)
    #         optimal_param1 = param1[optimal_indices]
    #         optimal_param2 = param2[optimal_indices]

    #         # save parameters 
    #         losses[:, :, j] = loss 
    #         phis_1[j] = optimal_param1
    #         phis_2[j] = optimal_param2


    #     # write to file 
    #     d = dict()
    #     d['landscape'] = losses 
    #     d['theta_1'] = phis_1 
    #     d['theta_2'] = phis_2 

    #     with open(filename, 'wb') as fp: 
    #         pickle.dump(d, fp)
    #         print("Succesfully saved")
        
    #     fp.close()

