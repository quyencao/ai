import helper
import pso_ann
import pso_flnn
import ga_flnn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

filename1 = 'data_resource_usage_fiveMinutes_6176858948.csv'
filename2 = 'data_resource_usage_twoMinutes_6176858948.csv'
filename3 = 'data_resource_usage_tenMinutes_6176858948.csv'
filename4 = 'Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv'

df = pd.read_csv(filename4, header=None, index_col=False, usecols=[0, 1])
df.dropna(inplace=True)

dataset_original = df.values

X, y = helper.process_data(dataset_original, helper.chebyshev_data, helper.transform, helper.sliding_data, 3, 4)

train_size = int(X.shape[0] * 0.8)
X_train, y_train, X_test, y_test = X[:train_size, :], y[:train_size, :], X[train_size:, :], y[train_size:, :]

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(dataset_original[:, 0])

p = pso_flnn.PSO()
# p = ga_flnn.Population()
best = p.train(X_train, y_train, epochs = 300)

y_pred = best.predict(X_test)

y_pred_inverted = scaler.inverse_transform(y_pred)
y_test_inverted = scaler.inverse_transform(y_test)

print("MAE: {0}".format(mean_absolute_error(y_pred_inverted, y_test_inverted)))

print(y_test_inverted)
print(y_pred_inverted)
print(dataset_original)

plt.plot(y_test_inverted[0, :], color='blue')
plt.plot(y_pred_inverted[0, :], color='red')
plt.show()

