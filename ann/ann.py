import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import helper

def init_parameters(n_inputs, n_hidden, n_outputs):
    W1 = np.random.randn(n_hidden, n_inputs)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_outputs, n_hidden)
    b2 = np.zeros((n_outputs, 1))
    
    return W1, b1, W2, b2

def init_momentum_parameters(W1, b1, W2, b2):
    vdW1 = np.zeros(W1.shape)
    vdb1 = np.zeros(b1.shape)
    
    vdW2 = np.zeros(W2.shape)
    vdb2 = np.zeros(b2.shape)
    
    return vdW1, vdb1, vdW2, vdb2

def tanh(x):
    e_plus = np.exp(x)
    e_minus = np.exp(-x)
    
    return (e_plus - e_minus) / (e_plus + e_minus)

def tanh_backward(x):
    return 1 - np.square(x)

def relu(x):
    a = np.maximum(0,x)
    return a

def relu_backward(x):
    dx = np.ones(x.shape)
    dx[x <= 0] = 0
    return dx

def feed_forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = tanh(Z2)
    
    return A2, Z2, A1, Z1

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

def model(X, y, X_valid, y_valid, X_test, y_test, n_hidden, learning_rate = 0.05, epochs = 1000, batch_size = 64, beta = 0.9):
    
    seed = 1
    
    n_inputs = X.shape[0]
    n_outputs = y.shape[0]
    
    W1, b1, W2, b2 = init_parameters(n_inputs, n_hidden, n_outputs)
    
    vdW1, vdb1, vdW2, vdb2 = init_momentum_parameters(W1, b1, W2, b2)
    
    train_rmse = []
    test_rmse = []
    
    for e in range(epochs):
        
        seed += 1
        
        mini_batches = random_mini_batches(X, y, mini_batch_size = batch_size, seed = seed)
        
        for mini_batch in mini_batches:
            X_batch, y_batch = mini_batch
            
            m = X_batch.shape[1]
            
            # Forward
            A2, Z2, A1, Z1 = feed_forward(X_batch, W1, b1, W2, b2)

            # Backpropagation
            dA2 = A2 - y_batch
            dZ2 = dA2 * tanh_backward(A2)

            db2 = 1./m * np.sum(dZ2, axis = 1, keepdims=True)
            dW2 = 1./m * np.dot(dZ2, A1.T)
            
            dA1 = np.dot(W2.T, dZ2)
            dZ1 = dA1 * tanh_backward(A1)
            
            db1 = 1./m * np.sum(dZ1, axis = 1, keepdims=True)
            dW1 = 1./m * np.dot(dZ1, X_batch.T)
            
            # Update W1, b1
            vdW1 = beta * vdW1 + (1 - beta) * dW1
            vdb1 = beta * vdb1 + (1 - beta) * db1
            
            W1 -= learning_rate * vdW1
            b1 -= learning_rate * vdb1
            
            # Update W2, b2
            vdW2 = beta * vdW2 + (1 - beta) * dW2
            vdb2 = beta * vdb2 + (1 - beta) * db2
            
            W2 -= learning_rate * vdW2
            b2 -= learning_rate * vdb2

        A_v, _, _, _ = feed_forward(X_train, W1, b1, W2, b2)

        mae_train = np.sqrt(mean_absolute_error(A_v, y_train))
            
        print("MAE Train: %.5f" % (mae_train))

    # Predict
    A_p, _, _, _ = feed_forward(X_test, W1, b1, W2, b2)
    
    return A_p


df = pd.read_csv('data_resource_usage_fiveMinutes_6176858948.csv', header=None, index_col=False, usecols=[3])
df.dropna(inplace=True)

dataset_original = df.values

X, y = helper.process_data(dataset_original, helper.power_data, helper.transform, helper.sliding_data, 3, 2)

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