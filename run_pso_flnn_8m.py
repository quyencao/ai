from ga_flnn.population import Population
from pso_flnn.pso import PSO
from flnn.flnn import FLNN
from ann.ann import ANN
import numpy as np
import pandas as pd

filename = 'data_resource_usage_8Minutes_6176858948.csv'
f = '8m'
foldername = 'data/'

# parameters
idx = (4160, 4160 + 1040)
c_couple = [(2, 2), (1,2, 1.2), (0.8, 2.0), (1.6, 0.6)]
sliding_windows = [2, 3, 5]
pop_sizes = [100, 150, 200]
cross_rates = [0.7, 0.8, 0.9]
mutate_rates = [0.01, 0.02, 0.03]
method_statistic = [0, 1, 2]
n_expanded = [2,3,4]
activations = [3, 2, 1, 0]

df = pd.read_csv(foldername + filename, header=None, index_col=False, usecols=[3, 4])
df.dropna(inplace=True)
dataset_original = df.values

for sliding in sliding_windows:
    for pop_size in pop_sizes:
        for cc in c_couple:
            for ms in method_statistic:
                for ne in n_expanded:
                    for activation in activations:
                        p = PSO(dataset_original, idx[0], idx[1], sliding, pop_size, cc[0], cc[1], ms, ne, activation, f)
                        p.train(epochs = 500)
