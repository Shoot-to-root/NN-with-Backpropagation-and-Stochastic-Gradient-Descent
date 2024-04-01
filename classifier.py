import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    s = np.maximum(0,x)
    return s

def init_params(n_input, n_hidden, n_hidden2, n_output):   
    # initialize weights and bias
    w1 = np.random.randn(n_hidden, n_input)*0.01
    b1 = np.zeros((n_hidden,1))
    w2 = np.random.randn(n_hidden2, n_hidden)*0.01
    b2 = np.zeros((n_hidden2,1))
    w3 = np.random.randn(n_output, n_hidden2)*0.01
    b3 = np.zeros((n_output,1))
    
    params = {"w1": w1,
              "b1": b1,
              "w2": w2,
              "b2": b2,
              "w3": w3,
              "b3": b3}
    
    return params  

def forward(X, params):
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]
    w3 = params["w3"]
    b3 = params["b3"]
    
    z1 = np.dot(w1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)
    
    # cache to calculate more easily when backpropagate
    cache = (z1, a1, w1, b1, z2, a2, w2, b2, z3, a3, w3, b3)
    
    return a3, cache

def backward(X, Y, cache):
    m = X.shape[1]
    (z1, a1, w1, b1, z2, a2, w2, b2, z3, a3, w3, b3) = cache
    
    dz3 = 1./m * (a3 - Y)
    dw3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)
    
    da2 = np.dot(w3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dw2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)
    
    da1 = np.dot(w2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dw1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)
    
    gradients = {"dz3": dz3, "dw3": dw3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dw2": dw2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dw1": dw1, "db1": db1}
    
    return gradients

def sgd(params, gradients, lr):
    L = len(params) // 2 # number of layers 

    for k in range(L):
        params["w" + str(k+1)] = params["w" + str(k+1)] - lr * gradients["dw" + str(k+1)]
        params["b" + str(k+1)] = params["b" + str(k+1)] - lr * gradients["db" + str(k+1)]
        
    return params

def loss(a3, Y):
    # cross entropy
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1./m * np.nansum(logprobs)
    
    return loss


def predict(X, y, params):
    m = X.shape[1]
    predict = np.zeros((1,m), dtype = np.int32)
    
    a3, caches = forward(X, params)
    
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            predict[0,i] = 1
        else:
            predict[0,i] = 0

    print("Accuracy: " + str(np.mean((predict[0,:] == y[0,:]))))
    
    return predict

def gen_batches(X, Y, batch_size = 64, seed = 0):  
    np.random.seed(seed)           
    m = X.shape[1]
    batches = []
    
    # shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    n_batches = math.floor(m / batch_size)
    for k in range(0, n_batches):
        batch_X = shuffled_X[:,k*batch_size:(k+1)*batch_size]
        batch_Y = shuffled_Y[:,k*batch_size:(k+1)*batch_size]
        batch = (batch_X, batch_Y)
        batches.append(batch)
    # last batch
    if m % batch_size != 0:
        batch_X = shuffled_X[:,n_batches*batch_size:]
        batch_Y = shuffled_Y[:,n_batches*batch_size:]
        batch = (batch_X, batch_Y)
        batches.append(batch)
    
    return batches

def train(X, Y, lr = 0.005, batch_size = 64, epochs = 10000):
    costs = []
    seed = 10                       
    
    params = init_params(34, 24, 10, 1)
    
    for i in range(epochs):
        batches = gen_batches(X, Y, batch_size, seed)
        
        # training 
        for batch in batches:
            (batch_X, batch_Y) = batch
            a3, caches = forward(batch_X, params)
            cost = loss(a3, batch_Y)
            gradients = backward(batch_X, batch_Y, caches)
            params = sgd(params, gradients, lr)
            
        if i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
            costs.append(cost)
            
        # plot latent features
        """
        if i == 3:
            plt.scatter(a3[:,1],a3[:,2],a3[:,3],'r');
            plt.scatter(a3[:,1],a3[:,2],a3[:,3],'b');
            plt.title('3D feature 3 epoch');
            plt.legend('class 1','class 2');
            plt.show()
        elif i == 300:
            plt.scatter(a3[:,1],a3[:,2],a3[:,3],'r');
            plt.scatter(a3[:,1],a3[:,2],a3[:,3],'b');
            plt.title('3D feature 3 epoch');
            plt.legend('class 1','class 2');
            plt.show()
        """
    
    #print(np.mean(costs))
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Test cross-entropy error")
    plt.show()
    
    return params
