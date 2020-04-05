import numpy as np
import matplotlib.pyplot as plt
from decimal import *

getcontext().prec = 30

def f(x):                # logistic map
	mu = 3.8
	y = mu*x*(1-x)
	return y

m=2
N=100
T=5000
sigma=0.05              # how to take the vaue of sigma isn't very clear
eps = 0.4
r = 0.32

z = np.zeros([N*m,T+1], dtype = float)    # m = number of layers, N = number of oscillator per layer,  T is number of time steps 

# calculation of adjacency matrix

A = np.ones(N) - np.eye(N)
I = np.eye(N)
Ad = np.random.rand(m,m,N,N)

for i in range(m):
	for j in range(m):
		if(i==j):
			Ad[i][j] = A
		else:
			Ad[i][j] = I

Ad=Ad.reshape(m,m*N,N)


Adj = Ad[0,:,:]

for i in range(1,m):
	Adj=np.concatenate((Adj,Ad[i,:,:]),axis=1)

# print(Adj)                      # Adj is the adjacency matrix, can be verified by taking small values of the parameters

for i in range(m):
	for j in range(N):

		# getcontext().prec += 3
		z[N*i+j][0]=np.exp(-((N*i+j-N)**2)/2*sigma*sigma)          # calculating the initial conditions of the oscillators, equation mentioned in the paper
			                                                     # i have used separate initial conditions for both the layers, and this is what they have done in paper too, acc to me, not very clear 


for t in range(T):

	if t%100 == 0:
		print(t)

	for i in range(N*m):

		Adj_term = 0
		for j in range(m*N):
			Adj_term+=Adj[i][j]*(f(z[j][t]) - f(z[i][t]))

		# print(Adj_term.shape)

		z[i][t+1] = f(z[i][t]) + eps*Adj_term/(2*r*N+1)       # Adj_term is the contribution of all other oscillators 

fig = plt.figure()
l1, l2 = plt.plot(z[:,T], 'b-', z[:,0], 'r-')
fig.legend((l1, l2), ('theta_1', 'theta_2'), 'upper left')
fig.set_size_inches(18.5, 10.5)
# plt.xlabel('t')
plt.show()