import numpy as np 
from scipy.special import expit

hidden_0=50 # number of nodes of first hidden layer
hidden_1=500 # number of nodes of second hidden layer

# Set up cost function
def costs(x,y,w_a,w_b,seed_):
    np.random.seed(seed_) # insert random seed
    w0=np.random.randn(hidden_0, 784) # weight matrix of 1st hidden layer
    w1=np.random.randn(hidden_1, hidden_0) # weight matrix of 2nd hidden layer
    w2=np.random.randn(10, hidden_1) # weight matrix of output layer
    w2[5][250] = w_a # set value for weight w_250,5(2)
    w2[5][251] = w_b # set value for weight w_251,5(2)
    a0 = expit(w0 @ x.T) # output of 1st hidden layer
    a1 = expit(w1 @ a0) # output of 2nd hidden layer
    pred = expit(w2 @ a1) # output of final layer
    return np.mean(np.sum((y-pred)**2,axis=0)) # costs w.r.t w_a and w_b