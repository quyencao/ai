import numpy as np
import random
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import helper

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
    def __init__(self, dataset_original, sliding, n_particles=100, c1 = 2, c2 = 2, v_max = 1, v_min = -1):
        self.dataset_original = dataset_original
        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.v_min = v_min
        self.w_max = 0.9
        self.w_min = 0.4
        self.pathsave = 'results/'
        self.filenamesave = "sliding-{0}-pop_size_{1}-c1_{2}-c2_{3}".format(sliding, n_particles, c1, c2)
        self.scaler = MinMaxScaler(feature_range=(0,1)).fit(dataset_original[:, 0])
        self.sliding = sliding
        # self.c1_max = 2.5
        # self.c1_min = 0.5
        # self.c2_max = 2.5
        # self.c2_min = 0.5
    
    def processing_data(self):
        X, y = helper.process_data(self.dataset_original, helper.chebyshev_data, helper.transform, helper.sliding_data, self.sliding, 4)

        train_size = int(X.shape[0] * 0.8)
        X_train, y_train, X_test, y_test = X[:train_size, :], y[:train_size, :], X[train_size:, :], y[train_size:, :]

        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    
    def save_file_csv(self):
        t1 = np.concatenate( (self.y_test_inverse, self.y_pred_inverse), axis = 1)
        np.savetxt(self.pathsave + self.filenamesave + ".csv", t1, delimiter=",")

    def predict(self, best_particle):
        y_pred = best_particle.predict(self.X_test)

        self.y_pred_inverse = self.scaler.inverse_transform(y_pred).T
        self.y_test_inverse = self.scaler.inverse_transform(self.y_test).T

        self.score_test_MAE = mean_absolute_error(self.y_pred_inverse, self.y_test_inverse)
        self.score_test_RMSE = np.sqrt(mean_squared_error(self.y_pred_inverse, self.y_test_inverse))

        self.draw_predict()

        self.save_file_csv()

    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.y_test_inverse[:, 0])
        plt.plot(self.y_pred_inverse[:, 0])
        plt.title('Model predict')
        plt.ylabel('Real value')
        plt.xlabel('Point')
        plt.legend(['realY... Test Score RMSE= ' + str(self.score_test_RMSE) , 'predictY... Test Score MAE= '+ str(self.score_test_MAE)], loc='upper right')
        plt.savefig(self.pathsave + self.filenamesave + ".png")
        plt.close()

    def initialize_particles(self, n_x, n_y):
        particles = []
        for i in range(self.n_particles):
            p = Particle(n_x, n_y)
            particles.append(p)
        return particles

    def train(self, epochs=450):
        self.processing_data()
        X_train, y_train = self.X_train, self.y_train

        n_x = X_train.shape[0]
        n_y = y_train.shape[0]

        gbest = None
        gbest_fitness = -1
        gbest_particle = None

        particles = self.initialize_particles(n_x, n_y)

        for e in range(epochs):

            w = self.w_min + (epochs - e) / epochs * (self.w_max - self.w_min)

            # c1 = (self.c1_min - self.c1_max) * e / epochs + self.c1_max

            # c2 = (self.c2_max - self.c2_min) * e / epochs + self.c2_min

            avg_mae_train = 0

            for p in particles:
                fitness = p.compute_fitness(X_train, y_train)

                if fitness > p.get_best_fitness():
                    p.set_best_fitness(fitness)
                    p.set_pbest(p.get_x())

                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest = p.get_x()
                    gbest_particle = copy.deepcopy(p)

                avg_mae_train += p.get_mae(X_train, y_train)

            print("Epoch %.f: %.5f" % (e + 1, avg_mae_train / len(particles)))

            for p in particles:
                x = p.get_x()
                pbest = p.get_pbest()
                v_o = p.get_v()

                v_n = w * v_o + self.c1 * random.uniform(0, 1) * (pbest - x) + self.c2 * random.uniform(0, 1) * (gbest - x)

                v_n[v_n > self.v_max] = self.v_max
                v_n[v_n < self.v_min] = self.v_min

                x_n = x + v_n

                p.set_v(v_n)
                p.set_x(x_n)
        self.predict(gbest_particle)
        