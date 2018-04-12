import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import helper


def init_parameters(n_inputs, n_outputs):
    W = np.random.randn(n_outputs, n_inputs)
    b = np.zeros((n_outputs, 1))
    
    return W, b

def init_momentum_parameters(n_inputs, n_outputs):
    vdW = np.zeros((n_outputs, n_inputs))
    vdb = np.zeros((n_outputs, 1))
    
    return vdW, vdb

def tanh(x):
    e_plus = np.exp(x)
    e_minus = np.exp(-x)
    
    return (e_plus - e_minus) / (e_plus + e_minus)

def tanh_backward(x):
    return 1 - np.square(x)

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model(X, y, X_valid, y_valid, X_test, y_test, learning_rate = 0.05, epochs = 1000, batch_size = 64, beta = 0.9):
    
    seed = 1
    
    n_inputs = X.shape[0]
    n_outputs = y.shape[0]
    
    W, b = init_parameters(n_inputs, n_outputs)
    
    vdW, vdb = init_momentum_parameters(n_inputs, n_outputs)
    
    maes_train = []
    maes_valid = []
    
    for e in range(epochs):
        
        seed += 1
        
        mini_batches = random_mini_batches(X, y, mini_batch_size = batch_size, seed = seed)
        
        total_cost = 0
        
        for mini_batch in mini_batches:
            X_batch, y_batch = mini_batch
            
            m = X_batch.shape[1]
            
            # Forward
            Z = np.dot(W, X_batch) + b
            A = tanh(Z)

            # Backpropagation
            dA = A - y_batch
            dZ = dA * tanh_backward(A)

            db = 1./m * np.sum(dZ, axis = 1, keepdims=True)
            dW = 1./m * np.dot(dZ, X_batch.T)
            
            vdW = beta * vdW + (1 - beta) * dW
            vdb = beta * vdb + (1 - beta) * db

            W -= learning_rate * vdW
            b -= learning_rate * vdb

        Z_t = np.dot(W, X_train) + b
        A_t = tanh(Z_t)
            
        mae_train = mean_absolute_error(A_t, y_train)
        print("MAE Train: %.5f" % (mae_train))

        if e % 10 == 0:

            Z_p = np.dot(W, X_valid) + b
            A_p = tanh(Z_p)
            
            mae_valid = mean_absolute_error(A_p, y_valid)

            Z_t = np.dot(W, X_train) + b
            A_t = tanh(Z_t)
            
            mae_train = mean_absolute_error(A_t, y_train)

            maes_train.append(mae_train)
            maes_valid.append(mae_valid)


    plt.plot(maes_train, color='blue')
    plt.plot(maes_valid, color='red')
    plt.show()        
        
    # Predict
    Z_p = np.dot(W, X_test) + b
    A_p = tanh(Z_p)

    return A_p

df = pd.read_csv('data_resource_usage_fiveMinutes_6176858948.csv', header=None, index_col=False, usecols=[3])
df.dropna(inplace=True)

dataset_original = df.values

X, y = helper.process_data(dataset_original, helper.legendre_data, helper.transform, helper.sliding_data, 3, 4)

train_size = int(0.7 * X.shape[0])
valid_size = int(0.15 * X.shape[0])
X_train, y_train, X_valid, y_valid, X_test, y_test = X[:train_size, :], y[:train_size, :], X[train_size:train_size+valid_size, :], y[train_size:train_size+valid_size, :], X[train_size+valid_size:, :], y[train_size+valid_size:, :]

X_train = X_train.T
X_test = X_test.T
X_valid = X_valid.T
y_train = y_train.T
y_valid = y_valid.T
y_test = y_test.T

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(dataset_original)

y_pred = model(X_train, y_train, X_valid, y_valid, X_test, y_test, 9)

y_pred_inverted = scaler.inverse_transform(y_pred)
y_test_inverted = scaler.inverse_transform(y_test)

print("MAE: %.5f" % (mean_absolute_error(y_pred_inverted, y_test_inverted)))

# print(y_pred_inverted)
# print(y_test_inverted)
plt.plot(y_test_inverted[0, :], color='blue')
plt.plot(y_pred_inverted[0, :], color='red')
plt.show()