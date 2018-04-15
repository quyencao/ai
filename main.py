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

df = pd.read_csv(filename1, header=None, index_col=False, usecols=[3, 5])
df.dropna(inplace=True)

dataset_original = df.values

# parameters
# c_couple = [(1,2, 1.2), (2, 2), (0.8, 2.0), (1.6, 0.6)]
sliding_windows = [2, 3, 4, 5]
pop_sizes = [100, 150, 200, 250, 300]
cross_rates = [0.7, 0.75, 0.8, 0.85, 0.9]
mutate_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
method_statistic = [0, 1, 2]

# for sliding in sliding_windows:
#     for pop_size in pop_sizes:
#         for c in c_couple:
#             p = pso_flnn.PSO(dataset_original, sliding, pop_size, c[0], c[1])
#             p.train(epochs = 300)

for sliding in sliding_windows:
    for pop_size in pop_sizes:
        for cr in cross_rates:
            for mr in mutate_rates:
                for ms in method_statistic:
                    p = ga_flnn.Population(dataset_original, sliding, pop_size, cr, mr, ms)
                    p.train(epochs = 300)
            

