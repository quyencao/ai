import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import helper

def init_parameters(n_inputs, n_hidden, n_outputs):
    W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(2 / n_inputs)
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_outputs, n_hidden) * np.sqrt(2 / n_hidden)
    b2 = np.zeros((n_outputs, 1))
    
    return W1, b1, W2, b2

def init_adam_parameters(W1, b1, W2, b2):
    vdW1 = np.zeros(W1.shape)
    vdb1 = np.zeros(b1.shape)
    sdW1 = np.zeros(W1.shape)
    sdb1 = np.zeros(b1.shape)
    
    vdW2 = np.zeros(W2.shape)
    vdb2 = np.zeros(b2.shape)
    sdW2 = np.zeros(W2.shape)
    sdb2 = np.zeros(b2.shape)
    
    return vdW1, vdb1, sdW1, sdb1, vdW2, vdb2, sdW2, sdb2

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
    A1 = relu(Z1)
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

def model(X, y, X_valid, y_valid, X_test, y_test, n_hidden, learning_rate = 0.0001, epochs = 300, batch_size = 32, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    
    seed = 1
    
    n_inputs = X.shape[0]
    n_outputs = y.shape[0]
    
    W1, b1, W2, b2 = init_parameters(n_inputs, n_hidden, n_outputs)
    
    vdW1, vdb1, sdW1, sdb1, vdW2, vdb2, sdW2, sdb2 = init_adam_parameters(W1, b1, W2, b2)
    
    t = 0
    
    train_rmse = []
    test_rmse = []
    
    min_valid_mae = 100
    best_w = {}
    best_b = {}
    
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
            dZ1 = dA1 * relu_backward(Z1)
            
            db1 = 1./m * np.sum(dZ1, axis = 1, keepdims=True)
            dW1 = 1./m * np.dot(dZ1, X_batch.T)
            
            t += 1
            
            # Update W1, b1
            vdW1 = beta1 * vdW1 + (1 - beta1) * dW1
            vdb1 = beta1 * vdb1 + (1 - beta1) * db1
            
            vdW1_corrected = vdW1 / (1 - np.power(beta1, t))
            vdb1_corrected = vdb1 / (1 - np.power(beta1, t))

            sdW1 = beta2 * sdW1 + (1 - beta2) * np.power(dW1, 2)
            sdb1 = beta2 * sdb1 + (1 - beta2) * np.power(db1, 2)
            
            sdW1_corrected = sdW1 / (1 - np.power(beta2, t))
            sdb1_corrected = sdb1 / (1 - np.power(beta2, t))
            
            W1 -= learning_rate * vdW1_corrected / (np.sqrt(sdW1_corrected) + epsilon)
            b1 -= learning_rate * vdb1_corrected / (np.sqrt(sdb1_corrected) + epsilon)
            
            # Update W2, b2
            vdW2 = beta1 * vdW2 + (1 - beta1) * dW2
            vdb2 = beta1 * vdb2 + (1 - beta1) * db2
            
            vdW2_corrected = vdW2 / (1 - np.power(beta1, t))
            vdb2_corrected = vdb2 / (1 - np.power(beta1, t))

            sdW2 = beta2 * sdW2 + (1 - beta2) * np.power(dW2, 2)
            sdb2 = beta2 * sdb2 + (1 - beta2) * np.power(db2, 2)
            
            sdW2_corrected = sdW2 / (1 - np.power(beta2, t))
            sdb2_corrected = sdb2 / (1 - np.power(beta2, t))
            
            W2 -= learning_rate * vdW2_corrected / (np.sqrt(sdW2_corrected) + epsilon)
            b2 -= learning_rate * vdb2_corrected / (np.sqrt(sdb2_corrected) + epsilon)

            A_v, _, _, _ = feed_forward(X_valid, W1, b1, W2, b2)
            
            A_v_unnorm = scaler.inverse_transform(A_v)

            mae_valid = np.sqrt(mean_absolute_error(A_v_unnorm, y_valid))
            
            # print("Valid: %.5f" % (mae_valid))
            
            if mae_valid < min_valid_mae:
                mae_valid = min_valid_mae
                best_w['W1'], best_w['W2'] = W1, W2
                best_b['b1'], best_b['b2'] = b1, b2
                

    # Predict
    A_p, _, _, _ = feed_forward(X_test, best_w['W1'], best_b['b1'], best_w['W2'], best_b['b2'])
    
    return A_p


df = pd.read_csv('sinwave.csv', header=None, index_col=False, usecols=[0])
df.dropna(inplace=True)

dataset_original = df.values

X, y = helper.process_data(dataset_original, helper.power_data, helper.transform, helper.sliding_data, 2, 2)

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