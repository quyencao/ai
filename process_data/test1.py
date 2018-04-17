# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:45:25 2018

@author: Quyen Cao
"""

import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('data_resource_usage_fiveMinutes_6176858948.csv', header=None, usecols=[3])

values = df.values

class Model:
    def __init__(self, dataset_original, sliding, method_statistic):
        self.dataset_original = dataset_original
        self.sliding = sliding
        self.min_max_scaler = MinMaxScaler()
        self.dimension = dataset_original.shape[1]
        self.test_idx = self.dataset_original.shape[0] - self.sliding
        self.method_statistic = method_statistic
    
    def inverse_data(self, transform_data):
        self.min_max_scaler.fit_transform(self.dataset_original[:, [0]])
        
        return self.min_max_scaler.inverse_transform(transform_data)    
    
    def processing_data(self):
        dataset_original, test_idx, sliding, method_statistic = self.dataset_original, self.test_idx , self.sliding, self.method_statistic
        
        list_split = []        
        for i in range(self.dimension):
            list_split.append(dataset_original[:, i:i+1])
            
        list_transform = []
        for i in range(len(list_split)):
            list_transform.append(self.min_max_scaler.fit_transform(list_split[i]))
            
        features = len(list_transform)
        
        dataset_sliding = np.zeros((test_idx, 1))
        for i in range(len(list_transform)):
            for j in range(sliding):
                d = np.array(list_transform[i][j:test_idx + j])
                dataset_sliding = np.concatenate((dataset_sliding, d), axis = 1)
        dataset_sliding = dataset_sliding[:, 1:]
        
        dataset_y = copy.deepcopy(list_transform[0][sliding:]) 
        
        if method_statistic == 0:
            dataset_X = copy.deepcopy(dataset_sliding)
        elif method_statistic == 1:
            dataset_X = np.zeros((dataset_sliding.shape[0], 1))
            for i in range(features):    
                mean = np.reshape(np.mean(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                dataset_X = np.concatenate((dataset_X, mean), axis = 1)
            dataset_X = dataset_X[:, 1:]
        elif method_statistic == 2:
            dataset_X = np.zeros((dataset_sliding.shape[0], 1))
            for i in range(features):    
                min_X = np.reshape(np.amin(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                median_X = np.reshape(np.median(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                max_X = np.reshape(np.amax(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                dataset_X = np.concatenate((dataset_X, min_X, median_X, max_X), axis = 1)
            dataset_X = dataset_X[:, 1:]
    
    def power_polynomials(self, n = 2):
        expanded_results = np.zeros((self.dataset_original.shape[0], 1))
        
        for i in range(self.dimension):
            for j in range(2, n):
                expanded = np.power(self.dataset_original[:, [i]], j)
                
                expanded_results = np.concatenate((expanded_results, expanded), axis = 1)
        
        expanded_results = expanded_results[:, 1:]
    
        return expanded_results
    
    def chebyshev_polynomials(self, n):
        expanded_results = np.zeros((self.dataset_original.shape[0], 1))
    
        for i in range(self.dimension):
            c1 = np.ones((self.dataset_original.shape[0], 1))
            c2 = self.dataset_original[:, [i]]
            for j in range(2, n):
                c = 2 * self.dataset_original[:, [i]] * c2 - c1
                c1 = c2
                c2 = c
    
                expanded_results = np.concatenate((expanded_results, c), axis=1)
    
        return expanded_results[:, 1:]
        
    def processing_data_2(self):
        dataset_original, test_idx, sliding, method_statistic = self.dataset_original, self.test_idx , self.sliding, self.method_statistic
        
        list_split = []        
        for i in range(self.dimension):
            list_split.append(dataset_original[:, i:i+1])
        
        # Expanded
        expanded = self.chebyshev_polynomials(3)
        for i in range(expanded.shape[1]):
            list_split.append(expanded[:, i:i+1])
        
        list_transform = []
        for i in range(len(list_split)):
            list_transform.append(self.min_max_scaler.fit_transform(list_split[i]))
            
        features = len(list_transform)
        
        dataset_sliding = np.zeros((test_idx, 1))
        for i in range(len(list_transform)):
            for j in range(sliding):
                d = np.array(list_transform[i][j:test_idx + j])
                dataset_sliding = np.concatenate((dataset_sliding, d), axis = 1)
        dataset_sliding = dataset_sliding[:, 1:]
        
        dataset_y = copy.deepcopy(list_transform[0][sliding:]) 
        
        if method_statistic == 0:
            dataset_X = copy.deepcopy(dataset_sliding)
        elif method_statistic == 1:
            dataset_X = np.zeros((dataset_sliding.shape[0], 1))
            for i in range(features):    
                mean = np.reshape(np.mean(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                dataset_X = np.concatenate((dataset_X, mean), axis = 1)
            dataset_X = dataset_X[:, 1:]
        elif method_statistic == 2:
            dataset_X = np.zeros((dataset_sliding.shape[0], 1))
            for i in range(features):    
                min_X = np.reshape(np.amin(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                median_X = np.reshape(np.median(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                max_X = np.reshape(np.amax(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                dataset_X = np.concatenate((dataset_X, min_X, median_X, max_X), axis = 1)
            dataset_X = dataset_X[:, 1:]
        
        inverse = self.inverse_data(dataset_y)        
        
        train_size = int(dataset_X.shape[0] * 0.8)
        X_train, y_train, X_test, y_test = dataset_X[:train_size, :], dataset_y[:train_size, :], dataset_X[train_size:, :], dataset_y[train_size:, :]

        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        
    
            
m = Model(values, 4, 2)
# m.processing_data()
m.processing_data_2()