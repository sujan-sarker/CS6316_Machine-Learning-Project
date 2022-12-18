import scipy
from collections import Counter
from performance_metrics import *
from DecisionTree import *
from random import randrange
import numpy as np

class RandomForest:
    def __init__(self, max_depth, min_size, sample_size, n_trees, n_features):
        self.max_depth = max_depth
        self.min_size= min_size
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.n_features = n_features
        self.trees = list()
     
            
    def subsample(self, dataset, ratio):
        sample = list()
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    # Make a prediction with a list of bagged trees
    def bagging_predict(self, trees, row):
        predictions = [self.predict_row(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)
    
    def predict_row(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_row(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_row(node['right'], row)
            else:
                return node['right']

    # Random Forest Algorithm
    def fit(self, x_train, y_train):
        #trees = list()
        train_data = np.concatenate((x_train, y_train.reshape(-1,1)), axis = 1)
        for i in range(self.n_trees):
            sample = self.subsample(train_data, self.sample_size)
            decision_tree = DecisionTree(self.max_depth, self.min_size)
            sample = np.array(sample)
            x_sample = sample[:,:-1]
            y_sample = np.array(sample[:,-1], dtype='int')
            decision_tree.fit(x_sample, y_sample, self.n_features)
            tree = decision_tree.tree
            self.trees.append(tree)
        #return trees


            
    def predict(self, test_data):
        predictions = list()
        trees = self.trees
        predictions = [self.bagging_predict(trees, row) for row in test_data]
        return predictions
  