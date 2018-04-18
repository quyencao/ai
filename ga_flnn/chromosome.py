import numpy as np
import random
from sklearn.metrics import mean_absolute_error

class Chromosome:
    def __init__(self, n_x, n_y, x=None):
        self.n_x = n_x
        self.n_y = n_y
        self.dimensions = n_x * n_y + n_y
        self.fitness = -1
        if x is None:
            self.x = np.random.uniform(low=-1, high=1, size=(self.dimensions))
        else:
            self.x = x

    def get_x(self):
        return np.copy(self.x)

    def set_x(self, value):
        self.x = value

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

        # error = np.sum(np.abs(A - y))
        error = mean_absolute_error(A, y)
        fitness = 1. / (error + 1)

        return fitness

    def set_fitness(self, value):
        self.fitness = value

    def get_fitness(self):
        return self.fitness

    def crossover(self, other):
        my_x = self.get_x()
        other_x = other.get_x()
        
        """
            Arithmetic crossover
        """
        r = random.uniform(0,1)
        child1_x = r * my_x + (1 - r) * other_x
        child2_x = (1 - r) * my_x + r * other_x

        child1 = Chromosome(self.n_x, self.n_y, child1_x)
        child2 = Chromosome(self.n_x, self.n_y, child2_x)

        return child1, child2

    def mutate(self, mutate_rate):
        for i in range(self.dimensions):
            if random.uniform(0,1) < mutate_rate:
                """
                    Uniform mutation
                """
                index = np.random.randint(self.dimensions)
                self.x[index] = np.random.uniform(low=-1, high=1)

    def predict(self, X_test):
        w, b = self.x[:self.n_x * self.n_y].reshape((self.n_y, -1)), self.x[self.n_x * self.n_y:].reshape((self.n_y, 1))

        Z = np.dot(w, X_test) + b
        A = self.tanh(Z)

        return A
