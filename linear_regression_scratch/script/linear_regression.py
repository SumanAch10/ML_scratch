import numpy as np

class LinearRegression:
    """Linear regression trained via batch gradient descent."""
    # Constructor 
    def __init__(self,learning_rate:float = 0.01,n_iters:int = 1000):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for gradient descent updates.
        n_iters : int
            Number of gradient descent iterations.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.loss_history = []
    
    def predict(self,X):
        y_pred = np.dot(X,self.w) + self.b
        return y_pred 
    
    def fit(self,X,y):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []
        
        for iter_ in range(self.n_iters):
            y_pred = self.predict(X)
            
            total_loss = np.sum((y_pred - y)**2) / (2*len(y))
            self.loss_history.append(total_loss)
            
            dw = (np.dot(y_pred - y,X)) / len(y)
            db = (np.sum(y_pred-y)) / len(y)
            
            self.w -= self.learning_rate*dw
            self.b -= self.learning_rate*db

