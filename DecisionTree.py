import scipy
from collections import Counter
from performance_metrics import *
from random import randrange
import math

class DecisionTree:
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size= min_size
        self.tree = None
        
    def calculate_gini_index(self, groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            N = float(len(group))
            if N == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / N
                score += math.pow(p,2)
            gini += (1.0 - score) * (N / n_instances)
        return gini
    
    
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
        
    def get_split(self, dataset, n_features=None):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        
        if n_features!= None:
            features = list()
            while len(features) < n_features:
                index = randrange(len(dataset[0])-1)
                if index not in features:
                    features.append(index)
            for index in features:
                for row in dataset:
                    groups = self.test_split(index, row[index], dataset)
                    gini = self.calculate_gini_index(groups, class_values)
                    if gini < b_score:
                        b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        else:
            for index in range(len(dataset[0])-1):
                for row in dataset:
                    groups = self.test_split(index, row[index], dataset)
                    gini = self.calculate_gini_index(groups, class_values)
                    if gini < b_score:
                        b_index, b_value, b_score, b_groups = index, row[index], gini, groups
                        
        return {'index':b_index, 'value':b_value, 'groups':b_groups}
    
    
    def to_terminal(self,group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
    
    
    def split(self,node, depth, n_features=None):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left,n_features)
            self.split(node['left'], depth+1, n_features)
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right, n_features)
            self.split(node['right'], depth+1, n_features)
            
    def fit(self, x_train, y_train, n_features=None):
        train_data = np.concatenate((x_train, y_train.reshape(-1,1)), axis = 1)
        root = self.get_split(train_data, n_features)
        self.split(root, 1, n_features)
        self.tree = root
        
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))
            
            
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
            
    def predict(self, test_data):
        predictions = list()
        node = self.tree
        for row in test_data:
            prediction = self.predict_row(node, row)
            predictions.append(prediction)
        return predictions
        #y_true = [row[-1] for row in test_data]    
        #print(p)
        #print(predictions)
        #return calculate_performance(predictions, y_true, is_sigmoid=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    