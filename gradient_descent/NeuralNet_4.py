import numpy as np
import matplotlib.pyplot as plt
from NeuralNet_1 import X_train, y_train_oh
from NeuralNet_2 import hidden_0, hidden_1
from scipy.special import expit

# Store values of costs and weights in lists:
weights_2_5_250=[]
weights_2_5_251=[]
costs=[]

seed_= 135 # random seed
N=100

# Set up neural network:
class NeuralNetwork(object):
    def __init__(self, lr=0.01):
        self.lr=lr
        np.random.seed(seed_) # set random seed
        # Initialize weight matrices
        self.w0=np.random.randn(hidden_0, 784)
        self.w1=np.random.randn(hidden_1, hidden_0)
        self.w2=np.random.randn(10, hidden_1)
        self.w2[5][250] = start_a # set starting value for w_a
        self.w2[5][251] = start_b # set starting value for w_b
    
    def train(self, X, y):
        a0= expit(self.w0 @ X.T)
        a1= expit(self.w1 @ a0)
        pred= expit(self.w2 @ a1)
        # Partial derivatives of costs w.r.t the weights of the output layer:
        dw2= (pred - y.T)*pred*(1-pred) @ a1.T / len(X) # ...averaged over the sample size
        # Update weights:
        self.w2[5][250]=self.w2[5][250] - self.lr * dw2[5][250]
        self.w2[5][251]=self.w2[5][251] - self.lr * dw2[5][251]
        costs.append(self.cost(pred,y)) # append cost values to list
    
    def cost(self, pred, y):
        return np.mean(np.sum((y.T - pred)**2, axis=0))

# Initial values of w_a/w_b:
starting_points = [(-9,15),(-10.1,15),(-11,15)]

for j in starting_points:
    start_a,start_b=j
    model=NeuralNetwork(10) #set learning rate to 10
    for i in range(10000): #10,000 epochs
        model.train(X_train[0:N], y_train_oh[0:N])
        weights_2_5_250.append(model.w2[5][250]) # append weight values to list
        weights_2_5_251.append(model.w2[5][251]) # append weight values to list

# Create sublists of costs and weight values for each starting point:
costs = np.split(np.array(costs),3)
weights_2_5_250 = np.split(np.array(weights_2_5_250),3)
weights_2_5_251 = np.split(np.array(weights_2_5_251),3)