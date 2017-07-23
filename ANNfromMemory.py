import numpy as np
import time

#XOR toy data
x0 = np.array([1,1,1,1])
x1 = np.array([1,1,0,0])
x2 = np.array([1,0,1,0])
y = np.array([0,1,1,0])
A = np.vstack((x0,x1,x2))

#Hyperparameters
threshold = 0.000001
n_layers = 3

#Activation function and its derivative
def activation(x, deriv=False):
	if deriv == True:
		return(x*(1-x))
	else:
		return(1/(1+np.e**(-x)))

#Initialize random weights (mean = 0)
synapses ={}
bias={}
synapses[0] = 2*np.random.random((3,2)) - 1
synapses[1] = 2*np.random.random((2,1)) - 1
bias[0] = 0
bias[1] = 0

t0=time.clock()
#Lillicrap's Feedback Alignment: B is a fixed random matrix to be used in all backprop
FA=True
B = 2*np.random.random((2,1)) - 1

layer = {}
error = {}
delta = {}
for iteration in range(100000):

	#Feedforward
	#rows of layer are number of nodes, columns are different samples
	layer[0] = A
	for i in range(1,n_layers):
		layer[i] = activation(synapses[i-1].T.dot(layer[i-1]) + bias[i-1])

	#Backprop
	#Error
	error[2] = y-layer[2]
	if (iteration%1000==0):
		print(error[2])
		#Check if MSE is below set threshold
		if np.mean(0.5*(error[2]**2))<threshold:
			break

	#The derivative of the activation is a function of the activation output
	#this is very convenient and efficient
	#dC/dW=(y-y')*a*(1-a)
	delta[2] = error[2]*activation(layer[2],True)

	#How much did layer 1 contribute to the error?
	#FA = Feedback Alignment
	if FA==True:
		error[1] = B.dot(delta[2])
	else:
		error[1] = synapses[1].dot(delta[2])

	delta[1] = error[1]*activation(layer[1],True)

	#update
	for i in synapses.keys():
		synapses[i] += layer[i].dot(delta[i+1].T)
		bias[i] += delta[i+1]

t1=time.clock()
print('Final output:\n', layer[2])
print('Time:', t1-t0)
