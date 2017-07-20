import numpy as np

#XOR toy data
x0 = np.array([1,1,1,1])
x1 = np.array([1,1,0,0])
x2 = np.array([1,0,1,0])
y = np.array([0,1,1,0])
A = np.vstack((x0,x1,x2))

#Hyperparameters
threshold = 0.00001

#Activation function and its derivative
def activation(x, deriv=False):
	if deriv == True:
		return(x*(1-x))
	else:
		return(1/(1+np.e**(-x)))

#Initialize random weights (mean = 0)
synapses ={}
synapses[0] = 2*np.random.random((3,2)) - 1
synapses[1] = 2*np.random.random((2,1)) - 1

for iteration in range(100000):

	#Feedforward
	#rows of layer are number of nodes, columns are different samples
	layer_0 = A
	layer_1 = activation(synapses[0].T.dot(layer_0))
	layer_2 = activation(synapses[1].T.dot(layer_1))

	#Backprop
	#Error
	l2_error = y-layer_2 #this is the derivative of 1/2(y-layer_2)^2 with respect to the activation function
	if (iteration%1000==0):
		print(l2_error)
		#Check if MSE is below set threshold
		if np.mean(0.5*(l2_error**2))<threshold:
			break

	#multiply the derivative dC/da * da/dw to obtain dC/dw
	l2_delta = l2_error*activation(layer_2,True)

	#How much did layer 1 contribute to the error?
	l1_error = synapses[1].dot(l2_delta)
	l1_delta = l1_error*activation(layer_1,True)

	#update
	synapses[1] += layer_1.dot(l2_delta.T)
	synapses[0] += layer_0.dot(l1_delta.T)

print(layer_2)
