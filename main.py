import helper
import pso_ann
import pso_flnn
import ga_flnn
import numpy as np
import pandas as pd


filename1 = 'data_resource_usage_fiveMinutes_6176858948.csv'
filename2 = 'data_resource_usage_twoMinutes_6176858948.csv'
filename3 = 'data_resource_usage_tenMinutes_6176858948.csv'
filename4 = 'Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv'

df = pd.read_csv(filename1, header=None, index_col=False, usecols=[3])
df.dropna(inplace=True)

dataset_original = df.values

# parameters
sliding_windows = [2, 3, 5]
c_couple = [(1,2, 1.2), (2, 2), (0.8, 2.0), (1.6, 0.6)]
pop_sizes = [100, 200, 300, 400]
ga_cr_mr = [(0.9, 0.01), (0.85, 0.02), (0.8, 0.05)]

# for sliding in sliding_windows:
#     for pop_size in pop_sizes:
#         for c in c_couple:
#             p = pso_flnn.PSO(dataset_original, sliding, pop_size, c[0], c[1])
#             p.train(epochs = 300)

for sliding in sliding_windows:
    for pop_size in pop_sizes:
        for mc in ga_cr_mr:
            p = ga_flnn.Population(dataset_original, sliding, pop_size, mc[0], mc[1])
            p.train(epochs = 300)
            

