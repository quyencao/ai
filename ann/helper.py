import numpy as np
from sklearn.preprocessing import MinMaxScaler

def transform(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    trans_data = np.zeros((data.shape[0], 1))
    for i in range(data.shape[1]):
        trans_data = np.concatenate((trans_data, scaler.fit_transform(data[:, [i]])), axis = 1)
    trans_data = trans_data[:, 1:]
    return trans_data


def sliding_data(data, sliding, features):
    X = []
    y = []

    for i in range(data.shape[0] - sliding):
        row_x = []
        row_y = []

        for j in range(sliding):
            for k in range(data.shape[1]):
                row_x.append(data[i + j, k])
        X.append(row_x)

        for j in range(features):
            row_y.append(data[i + sliding, j])
        y.append(row_y)

    X = np.array(X)
    y = np.array(y)

    return X, y


def chebyshev_data(data, n):
    expanded = np.zeros((data.shape[0], 1))

    for i in range(data.shape[1]):
        c1 = np.ones((data.shape[0], 1))
        c2 = data[:, [i]]
        for j in range(2, n):
            c = 2 * data[:, [i]] * c2 - c1
            c1 = c2
            c2 = c

            expanded = np.concatenate((expanded, c), axis=1)

    return expanded[:, 1:]

def legendre_data(data, n):
    expanded = np.zeros((data.shape[0], 1))

    for i in range(data.shape[1]):
        c1 = np.ones((data.shape[0], 1))
        c2 = data[:, [i]]
        for j in range(2, n):
            c = ((2 * j + 1) * data[:, [i]] * c2 - j * c1) / (j + 1)
            c1 = c2
            c2 = c

            expanded = np.concatenate((expanded, c), axis=1)

    return expanded[:, 1:]

def laguerre_data(data, n):
    expanded = np.zeros((data.shape[0], 1))

    for i in range(data.shape[1]):
        c1 = np.ones((data.shape[0], 1))
        c2 = data[:, [i]]
        for j in range(2, n):
            c = ((2 * j + 1 - data[:, [i]]) * c2 - j * c1) / (j + 1)
            c1 = c2
            c2 = c

            expanded = np.concatenate((expanded, c), axis=1)

    return expanded[:, 1:]

def power_data(data, n):
    expanded = np.zeros((data.shape[0], 1))

    for i in range(data.shape[1]):
        for j in range(2, n):
            power = np.power(data[:, [i]], j)

            expanded = np.concatenate((expanded, power), axis=1)

    return expanded[:, 1:]


def process_data(ori_data, power_data, transform, sliding_data, sliding=3, n_expanded=3):
    if n_expanded > 2:
        expanded = power_data(ori_data, n_expanded)
        all_data = np.concatenate((ori_data, expanded), axis=1)
    else:
        all_data = np.copy(ori_data)
 
    all_data_tr = transform(all_data)

    X, y = sliding_data(all_data_tr, sliding, 1)

    return X, y