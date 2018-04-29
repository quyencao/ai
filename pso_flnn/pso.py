import numpy as np
import random
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from particle import Particle

class PSO:
    def __init__(self, dataset_original, train_idx, test_idx, sliding, n_particles=100, c1 = 2, c2 = 2, method_statistic = 0, n_expanded = 2, activation = 0, fs = '2m'):
        self.dataset_original = dataset_original[:test_idx+sliding, :]
        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        self.v_max = 1
        self.v_min = -1
        self.w_max = 0.9
        self.w_min = 0.4
        self.pathsave = '/home/ubuntu/quyencao/ai/results/' + fs + '/'
        # self.pathsave = 'results/'
        self.filenamesave = "{0}-pso_flnn_sliding_{1}-pop_size_{2}-c1_{3}-c2_{4}-method_statistic_{5}-n_expanded_{6}-activation_{7}".format(fs, sliding, n_particles, c1, c2, method_statistic, n_expanded, activation)
        self.min_max_scaler = MinMaxScaler()
        self.sliding = sliding
        self.dimension = dataset_original.shape[1]
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.method_statistic = method_statistic
        self.n_expanded = n_expanded
        self.activation = activation

    def inverse_data(self, transform_data):
        self.min_max_scaler.fit_transform(self.dataset_original[:, [0]])
        
        return self.min_max_scaler.inverse_transform(transform_data)

    def power_polynomials(self, n = 2):
        expanded_results = np.zeros((self.dataset_original.shape[0], 1))
        
        for i in range(self.dimension):
            for j in range(2, n+2):
                expanded = np.power(self.dataset_original[:, [i]], j)
                
                expanded_results = np.concatenate((expanded_results, expanded), axis = 1)
        
        expanded_results = expanded_results[:, 1:]
    
        return expanded_results
    
    def chebyshev_polynomials(self, n):
        expanded_results = np.zeros((self.dataset_original.shape[0], 1))
    
        for i in range(self.dimension):
            c1 = np.ones((self.dataset_original.shape[0], 1))
            c2 = self.dataset_original[:, [i]]
            for j in range(2, n+2):
                c = 2 * self.dataset_original[:, [i]] * c2 - c1
                c1 = c2
                c2 = c
    
                expanded_results = np.concatenate((expanded_results, c), axis=1)
    
        return expanded_results[:, 1:]

    def legendre_data(self, n):
        expanded = np.zeros((self.dataset_original.shape[0], 1))

        for i in range(self.dimension):
            c1 = np.ones((self.dataset_original.shape[0], 1))
            c2 = self.dataset_original[:, [i]]
            for j in range(2, n+2):
                c = ((2 * j + 1) * data[:, [i]] * c2 - j * c1) / (j + 1)
                c1 = c2
                c2 = c

                expanded = np.concatenate((expanded, c), axis=1)

        return expanded[:, 1:]
        
    def processing_data_2(self):
        dataset_original, train_idx, test_idx, sliding, method_statistic, n_expanded = self.dataset_original, self.train_idx, self.test_idx , self.sliding, self.method_statistic, self.n_expanded
        
        list_split = []        
        for i in range(self.dimension):
            list_split.append(dataset_original[:, i:i+1])
        
        # Expanded
        expanded = self.chebyshev_polynomials(n_expanded)
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

        self.y_pred_inverse = self.inverse_data(y_pred).T
        self.y_test_inverse = self.inverse_data(self.y_test).T

        self.score_test_MAE = mean_absolute_error(self.y_pred_inverse, self.y_test_inverse)
        self.score_test_RMSE = np.sqrt(mean_squared_error(self.y_pred_inverse, self.y_test_inverse))

        print(self.score_test_MAE)

        self.draw_predict()
        self.save_file_csv()

    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.y_test_inverse[:, 0], color='#009FFD')
        plt.plot(self.y_pred_inverse[:, 0], color='#FFA400')
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
        self.processing_data_2()
        
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
                fitness = p.compute_fitness(X_train, y_train, self.activation)

                if fitness > p.get_best_fitness():
                    p.set_best_fitness(fitness)
                    p.set_pbest(p.get_x())

                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest = p.get_x()
                    gbest_particle = copy.deepcopy(p)

                avg_mae_train += p.get_mae(X_train, y_train)

            # print("Epoch %.f: %.5f" % (e + 1, avg_mae_train / len(particles)))

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
        