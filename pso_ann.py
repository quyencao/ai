import numpy as np
import random
import copy

class Particle:
    def __init__(self, n_x, n_h, n_y):
        self.n_x = n_x
        self.n_y = n_y
        self.n_h = n_h
        self.n_dimentions = n_x * n_h + n_h + n_h * n_y + n_y
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
        n_x, n_h, n_y = self.n_x, self.n_h, self.n_y
        index1 = n_h * n_x
        index2 = index1 + n_h
        index3 = index2 + n_h * n_y
        index4 = index3 + n_y
        w1, b1, w2, b2 = self.x[:index1].reshape((n_h, n_x)), self.x[index1:index2].reshape((n_h, 1)), self.x[
                                                                                                       index2:index3].reshape(
            (n_y, n_h)), self.x[index3:index4].reshape((n_y, 1))

        Z1 = np.dot(w1, X) + b1
        A1 = self.tanh(Z1)
        Z2 = np.dot(w2, A1) + b2
        A2 = self.tanh(Z2)

        error = np.sum(np.abs(A2 - y))
        fitness = 1. / (error + 1)

        return fitness

    def get_mae(self, X, y):
        n_x, n_y = self.n_x, self.n_y
        w, b = self.x[:n_x * n_y].reshape((n_y, -1)), self.x[n_x * n_y:].reshape((n_y, 1))

        Z = np.dot(w, X) + b
        #         A = self.tanh(Z)

        return 1. / X.shape[1] * np.sum(np.abs(Z - y))

    def predict(self, X):
        n_x, n_h, n_y = self.n_x, self.n_h, self.n_y
        index1 = n_h * n_x
        index2 = index1 + n_h
        index3 = index2 + n_h * n_y
        index4 = index3 + n_y
        w1, b1, w2, b2 = self.x[:index1].reshape((n_h, n_x)), self.x[index1:index2].reshape((n_h, 1)), self.x[
                                                                                                       index2:index3].reshape(
            (n_y, n_h)), self.x[index3:index4].reshape((n_y, 1))
        Z1 = np.dot(w1, X) + b1
        A1 = self.tanh(Z1)
        Z2 = np.dot(w2, A1) + b2
        A2 = self.tanh(Z2)

        return A2


class PSO:
    def __init__(self, n_particles=100):
        self.n_particles = n_particles
        self.c1 = 2
        self.c2 = 2
        self.w = 1.2
        self.v_max = 1
        self.v_min = -1
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 1
        self.c2_max = 2.5
        self.c2_min = 1

    def initialize_particles(self, n_x, n_h, n_y):
        particles = []
        for i in range(self.n_particles):
            p = Particle(n_x, n_h, n_y)
            particles.append(p)
        return particles

    def train(self, X_train, y_train, epochs=200):
        n_x = X_train.shape[0]
        n_h = 9
        n_y = y_train.shape[0]

        gbest = None
        gbest_fitness = -1
        gbest_particle = None

        particles = self.initialize_particles(n_x, n_h, n_y)

        for e in range(epochs):

            w = self.w_min + (epochs - e) / epochs * (self.w_max - self.w_min)

            c1 = (self.c1_min - self.c1_max) * e / epochs + self.c1_max

            c2 = (self.c2_max - self.c2_min) * e / epochs + self.c2_min

            for p in particles:
                fitness = p.compute_fitness(X_train, y_train)

                if fitness > p.get_best_fitness():
                    p.set_best_fitness(fitness)
                    p.set_pbest(p.get_x())

                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest = p.get_x()
                    gbest_particle = copy.deepcopy(p)

            print("Epoch %.f: %.5f" % (e , gbest_fitness))
            # file nay pso voi ann
            for p in particles:
                x = p.get_x()
                pbest = p.get_pbest()
                v_o = p.get_v()

                v_n = w * v_o + c1 * random.uniform(0, 1) * (pbest - x) + c2 * random.uniform(0, 1) * (gbest - x)

                v_n[v_n > self.v_max] = self.v_max
                v_n[v_n < self.v_min] = self.v_min

                x_n = x + v_n

                p.set_v(v_n)
                p.set_x(x_n)

        return gbest_particle