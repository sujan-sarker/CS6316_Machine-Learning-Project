import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from performance_metrics import *


class AdaBoost:
    
    def __init__(self):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, x_train, y_train, M = 100):
        y_train = y_train * 2 - 1
        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        self.M = M

        # Iterate over M weak classifiers
        for m in range(0, M):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y_train)) * 1 / len(y_train)  
            else:
                
                w_i = self.update_weights(w_i, alpha_m, y_train, y_pred)
            
            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(x_train, y_train, sample_weight = w_i)
            y_pred = G_m.predict(x_train)
            
            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = self.compute_error(y_train, y_pred, w_i)
            self.training_errors.append(error_m)

            # (c) Compute alpha
            alpha_m = self.compute_weight(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)
        
    
    # Compute weak classifier's error rate
    def compute_error(self, y_train, y_pred, w_i):
        return (sum(w_i * (np.not_equal(y_train, y_pred)).astype(int)))/sum(w_i)
    
    # Compute weak classifier weight
    def compute_weight(self, error):
        return np.log((1 - error) / error)
    
    # Update weights after a boosting iteration
    def update_weights(self, w_i, alpha, y_train, y_pred):
        return w_i * np.exp(alpha * (np.not_equal(y_train, y_pred)).astype(int))
    
    def predict(self, x_test):

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(x_test)), columns = range(self.M)) 

        # Predict class label for each weak classifier, weighted by alpha_  # Calculate final predictionsm
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(x_test) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m
        #print(weak_preds)
        # Calculate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)
        
        y_pred = y_pred.tolist()
        y_pred = [(item+1)/2 for item in y_pred]
        return y_pred

