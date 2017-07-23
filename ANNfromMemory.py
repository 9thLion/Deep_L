import numpy as np

#XOR toy data
x0 = np.array([1,1,1,1])
x1 = np.array([1,1,0,0])
x2 = np.array([1,0,1,0])
y = np.array([0,1,1,0])
A = np.vstack((x0,x1,x2))

#Hyperparameters
threshold = 0.000001

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
bias1 = 0
bias2 = 0

#Lillicrap's Feedback Alignment: B is a fixed random matrix to be used in all backprop
FA=True
B = 2*np.random.random((2,1)) - 1
for iteration in range(100000):

	#Feedforward
	#rows of layer are number of nodes, columns are different samples
	layer_0 = A
	layer_1 = activation(synapses[0].T.dot(layer_0) + bias1)
	layer_2 = activation(synapses[1].T.dot(layer_1) + bias2)

	#Backprop
	#Error
	l2_error = y-layer_2
	if (iteration%1000==0):
		print(l2_error)
		#Check if MSE is below set threshold
		if np.mean(0.5*(l2_error**2))<threshold:
			break

	#The derivative of the activation is a function of the activation output
	#this is very convenient and efficient
	#dC/dW=(y-y')*a*(1-a)
	l2_delta = l2_error*activation(layer_2,True)

	#How much did layer 1 contribute to the error?
	#FA = Feedback Alignment
	if FA==True:
		l1_error = B.dot(l2_delta)
	else:
		l1_error = synapses[1].dot(l2_delta)

	l1_delta = l1_error*activation(layer_1,True)

	#update
	synapses[1] += layer_1.dot(l2_delta.T)
	synapses[0] += layer_0.dot(l1_delta.T)
	bias1 += l1_delta
	bias2 += l2_delta

print(layer_2)
