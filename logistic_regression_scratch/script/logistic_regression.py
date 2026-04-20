import numpy as np

class LogisticRegression:
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
    
    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return 1 / (1 + np.exp(-z))
    
    def predict(self,X):
        y_pred = self.predict_proba(X)
        return (y_pred > 0.5).astype(int)
        
    def fit(self,X,y):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []
        
        for iter_ in range(self.n_iters):
            epsilon = 1e-15
            y_pred = self.predict_proba(X)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            total_loss = -np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)) / (len(y))
            self.loss_history.append(total_loss)
            
            dw = (np.dot(y_pred - y,X)) / len(y)
            db = (np.sum(y_pred-y)) / len(y)
            
            self.w -= self.learning_rate*dw
            self.b -= self.learning_rate*db

