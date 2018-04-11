import helper
import pso_ann
import pso_flnn
import ga_flnn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv('data_resource_usage_fiveMinutes_6176858948.csv', header=None, index_col=False, usecols=[3])
df.dropna(inplace=True)

dataset_original = df.values

X, y = helper.process_data(dataset_original, helper.power_data, helper.transform, helper.sliding_data, 2, 4)

train_size = int(X.shape[0] * 0.8)
X_train, y_train, X_test, y_test = X[:train_size, :], y[:train_size, :], X[train_size:, :], y[train_size:, :]

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(dataset_original)

# pso = pso_flnn.PSO()
p = ga_flnn.Population()
best = p.train(X_train, y_train)

y_pred = best.predict(X_test)

y_pred_inverted = scaler.inverse_transform(y_pred)
y_test_inverted = scaler.inverse_transform(y_test)

print("MAE: %.5f" % (mean_absolute_error(y_pred_inverted, y_test_inverted)))

print(y_test_inverted)
print(y_pred_inverted)
print(dataset_original)
# print(y_test_inverted)
plt.plot(y_test_inverted[0, :], color='blue')
plt.plot(y_pred_inverted[0, :], color='red')
plt.show()

