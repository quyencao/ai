import numpy as np
import random
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class RNN:    
    def __init__(self, dataset_original, train_idx, test_idx, sliding, method_statistic = 0, activation = 0, fs = '2m'):
        self.dataset_original = dataset_original[:test_idx+sliding, :]
        # self.pathsave = '/home/ubuntu/quyencao/ai/results/' + fs + '/'
        self.pathsave = 'results/'
        self.filenamesave = "{0}-pso_flnn_sliding_{1}-method_statistic_{2}-activation_{3}".format(fs, sliding, method_statistic, activation)
        self.min_max_scaler = MinMaxScaler()
        self.sliding = sliding
        self.dimension = dataset_original.shape[1]
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.method_statistic = method_statistic
        self.activation = activation
    
    def processing_data_2(self):
        dataset_original, train_idx, test_idx, sliding, method_statistic = self.dataset_original, self.train_idx, self.test_idx , self.sliding, self.method_statistic
        
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
        
        # train_size = int(dataset_X.shape[0] * 0.8)
        X_train, y_train, X_test, y_test = dataset_X[:train_idx, :], dataset_y[:train_idx, :], dataset_X[train_idx:, :], dataset_y[train_idx:, :]

        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(X_train)
        print(X_test)

        print(X_train.shape)
        print(X_test.shape)