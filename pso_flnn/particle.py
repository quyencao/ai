import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Particle:
    def __init__(self, n_x, n_y):
        self.n_x = n_x
        self.n_y = n_y
        self.n_dimentions = n_x * n_y + n_y
        self.x = np.random.uniform(low=-1, high=1, size=(self.n_dimentions))
        self.pbest = np.copy(self.x)
        self.v = np.zeros((self.n_dimentions))
        self.best_fitness = -1

    def get_x(self):
        return np.copy(self.x)

    def set_x(self, value):
        self.x = value

    def get_pbest(self):
        return np.copy(self.pbest)

    def set_pbest(self, value):
        self.pbest = value

    def get_v(self):
        return np.copy(self.v)

    def set_v(self, value):
        self.v = value

    def get_best_fitness(self):
        return self.best_fitness

    def set_best_fitness(self, value):
        self.best_fitness = value

    def tanh(self, x):
        e_plus = np.exp(x)
        e_minus = np.exp(-x)
        return (e_plus - e_minus) / (e_plus + e_minus)
    
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def relu(self, x):
        zeros = np.zeros(x.shape)
        return np.maximum(zeros, x)

    def elu(self, x):
        exp_x = np.exp(x) - 1
        exp_x[x >= 0] = 0

        copy_x = np.copy(x)
        copy_x[x < 0] = 0

        return copy_x + exp_x

    def compute_fitness(self, X, y, activation = 0):
        w, b = self.x[:self.n_x * self.n_y].reshape((self.n_y, -1)), self.x[self.n_x * self.n_y:].reshape((self.n_y, 1))

        Z = np.dot(w, X) + b

        if activation == 0:
            A = Z
        elif activation == 1:
            A = self.tanh(Z)
        elif activation == 2:
            A = self.relu(Z)
        elif activation == 3:
            A = self.elu(Z)

        error = np.sum(np.abs(A - y))
        fitness = 1. / (error + 1)

        return fitness

    def predict(self, X):
        n_x, n_y = self.n_x, self.n_y
        w, b = self.x[:n_x * n_y].reshape((n_y, -1)), self.x[n_x * n_y:].reshape((n_y, 1))

        Z = np.dot(w, X) + b
        A = self.tanh(Z)

        return A

    def get_mae(self, X, y):
        n_x, n_y = self.n_x, self.n_y
        w, b = self.pbest[:n_x * n_y].reshape((n_y, -1)), self.pbest[n_x * n_y:].reshape((n_y, 1))

        Z = np.dot(w, X) + b
        A = self.tanh(Z)
        return mean_absolute_error(A, y)