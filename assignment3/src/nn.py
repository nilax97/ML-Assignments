import numpy as np
import sklearn
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

import time
import sys
import warnings

if not sys.warnoptions:
	warnings.simplefilter("ignore")
config = sys.argv[1];
path_train = sys.argv[2];
path_test = sys.argv[3];

def initialize(layers):
    N = len(layers);
    param = {};
    for i in range(1,N):
        param["W" + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01;
        param["b" + str(i)] = np.zeros((layers[i],1));
    return param;

def sigmoid(Z):
    A = 1/(1+np.exp(-Z));
    cache = Z;
    
    return A, cache;

def relu(Z):
    A = np.maximum(0,Z);   
    cache = Z;
    
    return A, cache;

def relu_backward(dA, cache):
    
    Z = cache;
    dZ = np.array(dA, copy=True);
    dZ[Z <= 0] = 0;
    
    return dZ;

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def model_forward(X, parameters, activation):

    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b' + str(l)],activation)
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
    caches.append(cache)
    assert(AL.shape[1] == X.shape[1])
            
    return AL, caches

def compute_cost(AL, Y):
    
    m = Y.shape[1]
    logprobs = np.multiply(np.log(AL), Y) + np.multiply((1 - Y), np.log(1 - AL))
    cost = -1/m*np.sum(logprobs)
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m*np.dot(dZ,A_prev.T)
    db =  1/m*(np.sum(dZ,axis=1, keepdims=True))
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db

def model_backward(AL, Y, caches, activation):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,activation)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for i in range(1,L+1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate*grads["dW"+str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate*grads["db"+str(i)]
        
    return parameters

def model(X, Y, layers, lr = 0.1, num_iter = 2500, print_cost = False, batch_size = 100, fn = "sigmoid", tol = 1e-4, var="fixed"):
    costs = np.zeros(num_iter+2);
    n_batch = int(X.shape[0]/batch_size);
    parameters = initialize(layers);
    for i in range(num_iter):
        cost = 0;
        for x in range(n_batch):
            start = x*batch_size;
            end = (x+1)*batch_size;
            if(x==n_batch-1):
                end = X.shape[0]-1;
            X_batch = X[start:end,:].T;
            Y_batch = Y[start:end,:].T;
        
            AL, values = model_forward(X_batch, parameters, activation=fn);
            cost += compute_cost(AL, Y_batch);
            grads = model_backward(AL, Y_batch, values, activation= fn);
            parameters = update_parameters(parameters, grads, lr);
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        costs[i+2] = cost;
        if(np.abs(costs[i+1]-costs[i])<tol and (costs[i+2] - costs[i])<tol and var=="variable"):
            print("Updating LR");
            lr = lr/5;
    return parameters;

file = open(config, "r");
lines = file.readlines()
inp = int(lines[0].strip());
out = int(lines[1].strip());
batch = int(lines[2].strip());
n_hid = int(lines[3].strip());
layers = np.zeros(n_hid + 2).astype(int);
layers[0] = inp;
hid = lines[4].strip().split();
layers[n_hid+1] = out;
for i in range(1, n_hid+1):
    layers[i] = int(hid[i-1]);
act = lines[5].strip();
var = lines[6].strip();

train_arr = np.genfromtxt(path_train,delimiter=',');
test_arr = np.genfromtxt(path_test,delimiter=',');

X_train = train_arr[:,0:inp].copy();
Y_train = train_arr[:,inp:].copy();
X_test = test_arr[:,0:inp].copy();
Y_test = test_arr[:,inp:].copy();

parameters = model(X_train, Y_train, layers, lr = 0.1, num_iter = 3000, print_cost = False, batch_size = batch, fn = act, tol = 1e-4, var=var);
true_preds = Y_train.argmax(axis = 1)
pred_y, values = model_forward(X_train.T, parameters, activation=act);
pred_y = pred_y.argmax(axis = 0);
print("Train Accuracy:",accuracy_score(true_preds,pred_y));

true_preds = Y_test.argmax(axis = 1)
pred_y, values = model_forward(X_test.T, parameters, activation=act);
pred_y = pred_y.argmax(axis = 0);
print("Test Accuracy:",accuracy_score(true_preds,pred_y));
print(confusion_matrix(true_preds,pred_y));








