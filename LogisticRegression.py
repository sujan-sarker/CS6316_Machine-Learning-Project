import numpy as np
from performance_metrics import *

def seed_numpy():
    np.random.seed(seed=88)
    
class Sigmoid:
    def __init__(self):
        None
    def forward(self, a):
        return 1.0/(1+np.exp(-a))
    def gradient(a):
        sigmoid_val = self.forward(a)
        return sigmoid_val*(1-sigmoid_val)
    
        

class LogisticRegression:
    def __init__(self, feature_size, learning_rate = 0.01, lambda_val = 0.01, bias=True, seed_val=88):
        seed_numpy()
        self.w = np.random.rand(feature_size)
        self.b = np.random.rand() * bias
        self.lambda_val = lambda_val
        self.learning_rate = learning_rate
        self.activation = Sigmoid()
        self.feature_size = feature_size
    
    def forward(self, input):
        y = np.dot(input,self.w) + self.b 
        return self.activation.forward(y)
    
    def standardize(self, X_tr):
        for i in range(self.feature_size):
            X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
        return X_tr
    
    def cross_entropy_cost(self, input, label, regularization=False):
        cost = 0
        if regularization:
            cost = 0.5*self.lambda_val*np.sum(self.w**2)
        output = self.forward(input)
        
        length = input.shape[0]
        out = (-label * np.log(output) - (1-label) * np.log(1-output))
        out[np.isnan(out)] = 0
#         print(out)
        j_w = np.mean(out) + cost
        return j_w
    
    def gradient(self, input, label, regularization=False):
        length = input.shape[0]
        
        d_j_b = (self.forward(input)-label)
        d_j_w = self.learning_rate * np.dot(input.T, d_j_b)/length
        d_j_b = self.learning_rate * np.sum(d_j_b)/length
        
        regularization_d_j_w = 0
        regularization_d_j_b = 0
        
        if regularization:
            regularization_d_j_w = self.learning_rate * self.lambda_val * self.w /length
            regularization_d_j_b = self.learning_rate * self.lambda_val * self.b /length
            
        d_j_w += regularization_d_j_w
        d_j_b += regularization_d_j_b
        
        return d_j_w, d_j_b
    def fit_model(self, epoch, input, label, regularization=True):
        
        losses = []
        best_epoch = -1
        input = self.standardize(input)
        
        best_loss = np.Inf
        W,B = None,None
        for i in range(epoch):
            loss = self.cross_entropy_cost(input, label, regularization)
            losses.append(loss)
            dw, db = self.gradient(input, label, regularization)
            self.w -= dw
            self.b -= db
            
            print('Epoch {}/{}: Training Loss: {}'.format(i+1, epoch, loss))
            if best_loss > loss:
                W = self.w
                B = self.b
                best_loss = loss
                best_epoch = i
        return losses, W,B
    
    def prediction(self, input):
        input = self.standardize(input)
        return self.forward(input)
    
    def performance(self, input, label):
        return calculate_performance(self.prediction(input), label, is_sigmoid=True)
            
            
        
        
        
        