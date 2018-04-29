from ga_flnn.population import Population
from pso_flnn.pso import PSO
from flnn.flnn import FLNN
from ann.ann import ANN
import numpy as np
import pandas as pd
from rnn.rnn import RNN

filename1 = 'data_resource_usage_3Minutes_6176858948.csv'
filename2 = 'data_resource_usage_5Minutes_6176858948.csv'
filename3 = 'data_resource_usage_8Minutes_6176858948.csv'
filename4 = 'data_resource_usage_10Minutes_6176858948.csv'

filenames = [filename1, filename2, filename3, filename4]
fses = ['3m', '5m', '8m', '10m']
foldername = 'data/'

# parameters
c_couple = [(2, 2), (1,2, 1.2), (0.8, 2.0), (1.6, 0.6)]
list_idx = [(10560, 10560 + 2640), (6336, 6336 + 1584), (4160, 4160 + 1040), (3168, 3168 + 792)]
# list_idx = [(11120, 11120 + 2780), (6672, 6672 + 1668), (4160, 4160 + 1040), (3336, 3336 + 834)]
sliding_windows = [2,3,5]
pop_sizes = [100, 150, 200]
cross_rates = [0.7, 0.8, 0.9]
mutate_rates = [0.01, 0.02, 0.03]
method_statistic = [0, 1, 2]
n_expanded = [2,3,4]
activations = [3, 2, 1, 0]

"""
    RUN GA - FLNN
"""
# for idx, filename in enumerate(filenames):
#     df = pd.read_csv(foldername + filename, header=None, index_col=False, usecols=[3, 4])
#     df.dropna(inplace=True)
#     dataset_original = df.values

#     fs = fses[idx]

#     for sliding in sliding_windows:
#         for pop_size in pop_sizes:
#             for cr in cross_rates:
#                 for mr in mutate_rates:
#                     for ms in method_statistic:
#                         for ne in n_expanded:
#                             for activation in activations:
#                                 p = Population(dataset_original, sliding, pop_size, cr, mr, ms, ne, activation, fs)
#                                 p.train(epochs = 500)

"""
    RUN PSO - FLNN
"""
# for idx, filename in enumerate(filenames):
#     df = pd.read_csv(foldername + filename1, header=None, index_col=False, usecols=[3, 4])
#     df.dropna(inplace=True)
#     dataset_original = df.values

#     fs = fses[0]

#     idxs = list_idx[0]

#     for sliding in sliding_windows:
#         for pop_size in pop_sizes:
#             for cc in c_couple:
#                 for ms in method_statistic:
#                     for ne in n_expanded:
#                         for activation in activations:
#                             p = PSO(dataset_original, idxs[0], idxs[1], sliding, pop_size, cc[0], cc[1], ms, ne, activation, fs)
#                             p.train(epochs = 500)

"""
    RUN FLNN
"""
# for idx, filename in enumerate(filenames):
#     df = pd.read_csv(foldername + filename, header=None, index_col=False, usecols=[3, 4])
#     df.dropna(inplace=True)
#     dataset_original = df.values

#     fs = fses[idx]
#     idxs = list_idx[idx]

#     for sliding in sliding_windows:
#         for ms in method_statistic:
#             for ne in n_expanded:
#                 for activation in activations:
#                     p = FLNN(dataset_original, idxs[0], idxs[1], sliding, ms, ne, activation, fs)
#                     p.train(epochs = 1000)
"""
    RUN ANN
"""
# for idx, filename in enumerate(filenames):
    # df = pd.read_csv(foldername + filename, header=None, index_col=False, usecols=[3, 4])
    # df.dropna(inplace=True)
    # dataset_original = df.values

#     fs = fses[1]
#     idxs = list_idx[1]

#     for sliding in sliding_windows:
#         for ms in method_statistic:
#             for activation in activations:
#                 p = ANN(dataset_original, idxs[0], idxs[1], sliding, ms, activation, fs)
#                 p.train(epochs = 1000)


df = pd.read_csv(foldername + filename1, header=None, index_col=False, usecols=[3, 4])
df.dropna(inplace=True)
dataset_original = df.values
rnn = RNN(dataset_original, list_idx[0][0], list_idx[0][1], 3, 0, 0, fses[0])

rnn.processing_data_2()