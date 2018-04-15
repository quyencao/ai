import numpy as np
import random
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import helper

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

        error = np.sum(np.abs(A - y))
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


class Population:
    def __init__(self, dataset_original, sliding = 2, pop_size=100, crossover_rate = 0.9, mutate_rate = 0.01, method_statistic = 2, n_expanded = 2, activation=0):
        self.dataset_original = dataset_original
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.best_fitness = -1
        self.best_chromosome = None
        self.pathsave = 'results/'
        self.filenamesave = "ga_flnn_sliding_{0}-pop_size_{1}-crossover_rate_{2}-mutate_rate_{3}-method_statistic_{4}-n_expanded_{5}-activation_{6}".format(sliding, pop_size, crossover_rate, mutate_rate, method_statistic, n_expanded, activation)
        self.min_max_scaler = MinMaxScaler()
        self.sliding = sliding
        self.dimension = dataset_original.shape[1]
        self.test_idx = self.dataset_original.shape[0] - self.sliding
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

        for i in range(data.shape[1]):
            c1 = np.ones((self.dataset_original.shape[0], 1))
            c2 = self.dataset_original[:, [i]]
            for j in range(2, n+2):
                c = ((2 * j + 1) * data[:, [i]] * c2 - j * c1) / (j + 1)
                c1 = c2
                c2 = c

                expanded = np.concatenate((expanded, c), axis=1)

        return expanded[:, 1:]
        
    def processing_data_2(self):
        dataset_original, test_idx, sliding, method_statistic, n_expanded = self.dataset_original, self.test_idx , self.sliding, self.method_statistic, self.n_expanded
        
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

        print(self.X_train.shape)

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
    
    def save_file_csv(self):
        t1 = np.concatenate( (self.y_test_inverse, self.y_pred_inverse), axis = 1)
        np.savetxt(self.pathsave + self.filenamesave + ".csv", t1, delimiter=",")

    def predict(self):
        y_pred = self.best_chromosome.predict(self.X_test)

        self.y_pred_inverse = self.inverse_data(y_pred).T
        self.y_test_inverse = self.inverse_data(self.y_test).T

        self.score_test_MAE = mean_absolute_error(self.y_pred_inverse, self.y_test_inverse)
        self.score_test_RMSE = np.sqrt(mean_squared_error(self.y_pred_inverse, self.y_test_inverse))

        self.draw_predict()
        self.save_file_csv()

    def init_population(self, X, y):
        n_x, n_y = X.shape[0], y.shape[0]
        population = []
        for i in range(self.pop_size):
            c = Chromosome(n_x, n_y)
            f = c.compute_fitness(X, y, self.activation)
            c.set_fitness(f)

            population.append(c)

            if f > self.best_fitness:
                self.best_fitness = f
                self.best_chromosome = copy.deepcopy(c)

        self.population = population

    def tournament_selection(self, fitnesses):
        k = 10
        chromosomes_fitness = []
        chromosomes_indices = []

        for i in range(k):
            index = np.random.randint(len(fitnesses))
            chromosomes_indices.append(index)
            chromosomes_fitness.append(fitnesses[index])

        chromosomes_fitness = np.array(chromosomes_fitness)

        best_chromosome_index = chromosomes_fitness.argsort()[-1]

        return chromosomes_indices[best_chromosome_index]

    def train(self, epochs=200):
        self.processing_data_2()

        X, y = self.X_train, self.y_train

        self.init_population(X, y)

        for e in range(epochs):

            print(self.best_fitness)

            fitnesses = np.array([p.get_fitness() for p in self.population])

            sorted_fitness = np.argsort(-1 * fitnesses)

            population = np.array(self.population)

            sorted_population = population[sorted_fitness]

            # Select 40% top produce offspring
            n = int(self.pop_size * 0.4)
            sub_population = sorted_population[:n]
            sub_fitnesses = fitnesses[sorted_fitness[:n]]

            next_population = []

            # keep 10% to next population
            next_population.extend(sorted_population[:int(0.05 * self.pop_size)])
            next_population.extend(sorted_population[-int(0.01 * self.pop_size):])

            while (len(next_population) < self.pop_size):

                parent1 = sub_population[self.tournament_selection(sub_fitnesses)]
                parent2 = sub_population[self.tournament_selection(sub_fitnesses)]

                possiblyCrossed = [parent1, parent2]

                if random.uniform(0, 1) < self.crossover_rate:
                    child1, child2 = parent1.crossover(parent2)

                    possiblyCrossed = [child1, child2]

                for chromosome in possiblyCrossed:
                    chromosome.mutate(self.mutate_rate)

                for chromosome in possiblyCrossed:
                    f = chromosome.compute_fitness(X, y, self.activation)

                    chromosome.set_fitness(f)

                    if f > self.best_fitness:
                        self.best_fitness = f
                        self.best_chromosome = copy.deepcopy(chromosome)

                    next_population.append(chromosome)
            self.population = next_population
        print("================DONE===================")
        self.predict()