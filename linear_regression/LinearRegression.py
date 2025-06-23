import numpy as np
from sklearn.metrics import r2_score

class LinearRegression:
 
    #create a constuctor
    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.cost = []
    
    #create function to fit the modelf
    def fit(self, X, y):
        #get the number of samples and features
        n_samples, n_features = X.shape

        #each features needs a weight
        self.weights = np.zeros(n_features)
        self.bias = 0

        #train model with gradient descent
        for _ in range(self.n_iters):
            #make predictions
            y_pred = self.predict(X)

            #calculate the error and add to the cost array
            error = self.mse(y, y_pred)
            self.cost.append(error)
            
            #calculate the cost funtion derivatives for weights and bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            #update the weights
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        
        return

    #create a function to calculate Mean squared error
    def mse(self, y_test, predictions):
        return np.mean((y_test - predictions)**2)

    #Function to make predictions
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    #Function to print the model stats
    def print_stats(self, y_test, predictions):
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias}")
        print(f"R square: {r2_score(y_test, predictions)}")