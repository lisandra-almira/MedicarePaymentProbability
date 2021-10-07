import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
import sys
from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
   
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes 
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate  
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                       self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                        self.no_of_hidden_nodes))
           
    
    def train(self):
        pass
    
    def run(self):
        pass
              

simple_network = NeuralNetwork(no_of_in_nodes = 3, 
                               no_of_out_nodes = 2, 
                               no_of_hidden_nodes = 4,
                               learning_rate = 0.1)
print(simple_network.weights_in_hidden)
print(simple_network.weights_hidden_out)

x_all=np.array(([[49, 0],[48,0],[62,1],[51,1],[16,0],[78,1],[54,1],[51,1],[47,0],[59,1],[55,1],[54,1],[42,0],[37,0],[49,0],[54,1],[49,0],[0.51979789,1],[45,0],[56,1]]),dtype=float)
y=np.array(([0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,1]),dtype=float)

x_all=x_all/np.max(x_all,axis=0)
y=y/100

X=np.split(x_all,[3])[0]
x_predicted=np.split(x_all,[3])[1]
print(X)
print(x_predicted)

def sigma(x):
    return 1 / (1 + np.exp(-x))

X = np.linspace(-5, 5, 100)


plt.plot(X, sigma(X),'b')
plt.xlabel('X')
plt.ylabel('Sigma(X)')
plt.title('Probability of Medicare Payment')

plt.grid()

plt.text(2.3, 0.84, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=16)


plt.show()

def forward(self,X):
    self.z=np.dot(X, self.W1)
    self.z2=self.sigmoid(self.z)
    self.z3=np.dot(self.z2,self.W2)
    o=self.sigmoid(self.z3)
    return o

def sigmoid(self,s):
    return 1/(1+np.exp(-s))

nn= neural_network()
o=nn.forward(X)

print("Predicted Output: \n"+str(o))
print("Actual Output: \n" + str(y))









