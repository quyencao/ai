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

    def compute_fitness(self, X, y):
        w, b = self.x[:self.n_x * self.n_y].reshape((self.n_y, -1)), self.x[self.n_x * self.n_y:].reshape((self.n_y, 1))

        Z = np.dot(w, X) + b
        A = self.tanh(Z)

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

        # child1_x = (my_x + other_x) / 2
        
        # pmax = np.ones(my_x.shape)
        # maxp1p2 = np.maximum(my_x, other_x)
        # child2_x = pmax * 0.4 + maxp1p2 * 0.6
        
        """
            Arithmetic crossover
        """
        r = random.uniform(0,1)
        child1_x = r * my_x + (1 - r) * other_x
        child2_x = (1 - r) * my_x + r * other_x

        """
            Heuristic crossover
        """
        # if self.get_fitness() > other.get_fitness():
        #     child1_x = my_x + random.uniform(0,1) * (my_x - other_x)
        #     child2_x = my_x
        # else:
        #     child1_x = other_x + random.uniform(0,1) * (other_x - my_x)
        #     child2_x = other_x

        child1 = Chromosome(self.n_x, self.n_y, child1_x)
        child2 = Chromosome(self.n_x, self.n_y, child2_x)

        return child1, child2

    def mutate(self, mutate_rate):
        for i in range(self.dimensions):
            if random.uniform(0,1) < mutate_rate:
                """
                    Uniform mutation
                """
                # index = np.random.randint(self.dimensions)
                # self.x[index] = np.random.uniform(low=-1, high=1)

                """
                    Boundary mutation
                """
                index = np.random.randint(self.dimensions)
                r = random.uniform(0,1)
                if r > 0.5:
                    self.x[index] = 1
                else:
                    self.x[index] = -1

    def predict(self, X_test):
        w, b = self.x[:self.n_x * self.n_y].reshape((self.n_y, -1)), self.x[self.n_x * self.n_y:].reshape((self.n_y, 1))

        Z = np.dot(w, X_test) + b
        A = self.tanh(Z)

        return A


class Population:
    def __init__(self, dataset_original, sliding = 2, pop_size=100, crossover_rate = 0.9, mutate_rate = 0.01):
        self.dataset_original = dataset_original
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.best_fitness = -1
        self.best_chromosome = None
        self.pathsave = 'results/'
        self.filenamesave = "ga_flnn_sliding_{0}-pop_size_{1}-crossover_rate_{2}-mutate_rate_{3}".format(sliding, pop_size, crossover_rate, mutate_rate)
        self.scaler = MinMaxScaler(feature_range=(0,1)).fit(dataset_original[:, 0])
        self.sliding = sliding

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

        self.y_pred_inverse = self.scaler.inverse_transform(y_pred).T
        self.y_test_inverse = self.scaler.inverse_transform(self.y_test).T

        self.score_test_MAE = mean_absolute_error(self.y_pred_inverse, self.y_test_inverse)
        self.score_test_RMSE = np.sqrt(mean_squared_error(self.y_pred_inverse, self.y_test_inverse))

        self.draw_predict()
        self.save_file_csv()

    def init_population(self, X, y):
        n_x, n_y = X.shape[0], y.shape[0]
        population = []
        for i in range(self.pop_size):
            c = Chromosome(n_x, n_y)
            f = c.compute_fitness(X, y)
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
        self.processing_data()

        X, y = self.X_train, self.y_train

        self.init_population(X, y)

        for e in range(epochs):
            print(self.best_fitness)

            fitnesses = np.array([p.get_fitness() for p in self.population])

            next_population = []

            while (len(next_population) < self.pop_size):

                parent1 = self.population[self.tournament_selection(fitnesses)]
                parent2 = self.population[self.tournament_selection(fitnesses)]

                possiblyCrossed = [parent1, parent2]

                if random.uniform(0, 1) < self.crossover_rate:
                    child1, child2 = parent1.crossover(parent2)

                    possiblyCrossed = [child1, child2]

                for chromosome in possiblyCrossed:
                    chromosome.mutate(self.mutate_rate)

                for chromosome in possiblyCrossed:
                    f = chromosome.compute_fitness(X, y)

                    chromosome.set_fitness(f)

                    if f > self.best_fitness:
                        self.best_fitness = f
                        self.best_chromosome = copy.deepcopy(chromosome)

                    next_population.append(chromosome)
            self.population = next_population

        self.predict()