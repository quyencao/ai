import numpy as np
import random
import copy
from sklearn.metrics import mean_absolute_error

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
        return 1. / (1 + np.exp(-x))

    def compute_fitness(self, X, y):
        w, b = self.x[:self.n_x * self.n_y].reshape((self.n_y, -1)), self.x[self.n_x * self.n_y:].reshape((self.n_y, 1))

        Z = np.dot(w, X) + b
        A = self.tanh(Z)

        # error = 1. / X.shape[1] * np.sum(np.abs(A - y))
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


class PSO:
    def __init__(self, n_particles=100):
        self.n_particles = n_particles
        self.c1 = 2
        self.c2 = 2
        self.v_max = 1
        self.v_min = -1
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2
        self.c1_min = 0.5
        self.c2_max = 2
        self.c2_min = 0.5

    def initialize_particles(self, n_x, n_y):
        particles = []
        for i in range(self.n_particles):
            p = Particle(n_x, n_y)
            particles.append(p)
        return particles

    def train(self, X_train, y_train, epochs=200):
        n_x = X_train.shape[0]
        n_y = y_train.shape[0]

        gbest = None
        gbest_fitness = -1
        gbest_particle = None

        particles = self.initialize_particles(n_x, n_y)

        for e in range(epochs):

            w = self.w_min + (epochs - e) / epochs * (self.w_max - self.w_min)

            c1 = (self.c1_min - self.c1_max) * e / epochs + self.c1_max

            c2 = (self.c2_max - self.c2_min) * e / epochs + self.c2_min

            avg_mae = 0

            for p in particles:
                fitness = p.compute_fitness(X_train, y_train)

                if fitness > p.get_best_fitness():
                    p.set_best_fitness(fitness)
                    p.set_pbest(p.get_x())

                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest = p.get_x()
                    gbest_particle = copy.deepcopy(p)

                avg_mae += p.get_mae(X_train, y_train)

            print("Epoch %.f: %.5f" % (e + 1, avg_mae / len(particles)))

            for p in particles:
                x = p.get_x()
                pbest = p.get_pbest()
                v_o = p.get_v()

                v_n = w * v_o + c1 * random.random() * (pbest - x) + c2 * random.random() * (gbest - x)

                v_n[v_n > self.v_max] = self.v_max
                v_n[v_n < self.v_min] = self.v_min

                x_n = x + v_n

                p.set_v(v_n)
                p.set_x(x_n)
        return gbest_particle