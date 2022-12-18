import numpy as np
from performance_metrics import *

def seed_numpy():
    np.random.seed(seed=88)

class SVM:
    def __init__(self, feature_size, learning_rate = 1e-2,reg_strength=2 ):
        self.W = np.random.rand(feature_size+1)
        self.reg_strength = reg_strength # regularization strength
        self.learning_rate = learning_rate
        
        self.feature_size = feature_size
        
        
    def compute_cost(self, X, Y):
        N = len(Y)
        distances = 1 - Y * (np.dot(X, self.W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = self.reg_strength * (np.sum(distances) / N)
        cost = 0.5 * np.dot(self.W, self.W) + hinge_loss
        return cost
    
    def standardize(self, X_tr):
        for i in range(self.feature_size):
            X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
        return X_tr
    
    def calculate_cost_gradient(self, X_batch, Y_batch):
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch], dtype='int')
#             X_batch = np.array([X_batch])
        distance = 1 - (Y_batch * np.dot(X_batch, self.W))
        dw = np.zeros(len(self.W))
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = self.W
            else:
                di = self.W - (self.reg_strength * Y_batch[ind] * X_batch[ind])
            dw += di
        dw = dw/len(Y_batch)  # average
        return dw
    
    def sgd(self,features, outputs, max_epochs = 100):
        
        features = self.standardize(features)
        outputs = self.relabel_data(outputs)
        N = features.shape[0]
        features = np.hstack((features,np.ones((features.shape[0],1))))
#         weights = np.zeros(features.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01
        
        costs = []
        
        best_w = None
        best_epoch = None
        for epoch in range(1, max_epochs):
            X, Y = features, outputs
            ascent = self.calculate_cost_gradient(X, Y)
            self.W -= (self.learning_rate * ascent)
        
        
            cost = self.compute_cost(features, outputs)
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            if prev_cost > cost:
                best_w = self.W
                best_epoch = epoch
            nth += 1
                
            costs.append(cost)
        return best_w, epoch, costs
    
    def relabel_data(self, Y):
        Y = 2*Y - 1
        return Y
    
    def label_data(self, Y):
        Y = np.array((Y + 1)/2, dtype='int')
        return Y
        
        
    def score(self, X_test, Y_test):
        X_test = self.standardize(X_test)
        X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))
#         Y_test = self.relabel_data(Y_test)
        
        y_pred = np.sign(np.dot(X_test, self.W))
        
        
        y_pred = self.label_data(y_pred)
        return calculate_performance(y_pred, Y_test, is_sigmoid=True)