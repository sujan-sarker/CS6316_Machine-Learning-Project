import scipy
from collections import Counter
from performance_metrics import *
import numpy as np

class KNN:
    def __init__(self, feature_size):
        self.feature_size = feature_size
    
    def standardize(self, X_tr):
        for i in range(self.feature_size):
            X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
        return X_tr
    
    def get_neighbors(self, train, test_row, num_neighbors):
        dist = np.sqrt(np.sum((train-test_row)**2, axis=1))
        return np.argsort(dist)[:num_neighbors]
    
    
    def predict_class(self, train, train_label, test_row, num_neighbors):
        neighbors = self.get_neighbors(train, test_row, num_neighbors)
        labels = train_label[neighbors]
        prediction = 1 * (np.sum(labels)>len(labels)-np.sum(labels))
        return prediction
    
    def prediction(self, train, labels, test, test_labels, num_neighbor):
        pred = []
        train = self.standardize(train)
        test = self.standardize(test)
        for i in range(len(test)):
            pred.append(self.predict_class(train, labels, test[i], num_neighbor))
        pred = np.array(pred)
        perf = self.score(pred, test_labels)
        return pred, perf
        
        
    def score(self, X_test, y_test):
        return calculate_performance(X_test, y_test, is_sigmoid=True)