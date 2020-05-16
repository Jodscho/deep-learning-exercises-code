import random
from numpy import random as r
import math
import matplotlib.pyplot as plt


def relu(value): return 0 if value < 0 else value

def relu_der(value): return 0 if value <= 0 else 1 

def loss(y, y_hat):
	loss = 0
	for i in range(len(y_hat)):
		loss += (y[i] - y_hat[i])**2
	return loss


def loss_der(y, y_hat):
	loss = 0
	for i in range(len(y_hat)):
		loss += (y[i] - y_hat[i])
	return -2*loss

def init_array(i, j = 0):
	if j is 0:
		return [0 for k in range(i)]
	else:
		return [[0 for m in range(j)] for k in range(i)]

def nodes_to_drop_out(layer_size):
	return [random.uniform(0, 1) > 0.8 for i in range(layer_size)]

def	constant_initalization(val):
	v = []
	for i in range(2): v.append(val)
	v.append(0) # for the bias

	w = []
	for i in range(2):
		w.append([])
		for j in range(2):
			w[i].append(val)
		w[i].append(0) # for the bias

	return w, v

def xaviar_initalization():
	# pick from the following gaussian distribution: 
	# N~(0, 1/N)

	# for v, (2+1)/2 = 1.5, 1/1.5 = 0.6666
	v = []
	for i in range(2):
		v.append(r.normal(loc=0, scale=0.666))
	v.append(0) # for the bias

	# for w, (2+2)/2 = 2, 1/2 = 0.5
	w = []
	for i in range(2):
		w.append([])
		for j in range(2):
			w[i].append(r.normal(loc=0, scale=0.5))
		w[i].append(0) # for the bias

	return w, v

def rms_prop(w, v, dataset, delta, roh, iterations):

	x_axis = []
	y_axis = []

	r_v = init_array(len(v))
	r_w = init_array(len(w), len(w[0]))
	mu = 0.1 # no mu given in task description

	for epoch in range(20):

		print("--- START EPOCH {} ----").format(epoch)

		x_axis.append(epoch)

		gradient_w, gradient_v = backpropagation_dataset(v, w, dataset, y_axis)
		#print("gradient_w: {}").format(gradient_w)
		#print("gradient_v: {}").format(gradient_v)



		for i in range(len(v)):
			r_v[i] = roh * r_v[i] + (1 - roh)*(v[i] * v[i])
			mu_tilde = mu / (delta + math.sqrt(r_v[i]))
			v[i] = v[i] - mu_tilde * gradient_v[i]

		for i in range(len(w)):
			for j in range(len(w[0])):
				r_w[i][j] = roh * r_w[i][j] + (1 - roh)*(w[i][j] * w[i][j])
				mu_tilde = mu / (delta + math.sqrt(r_w[i][j]))
				w[i][j] = w[i][j] - mu_tilde * gradient_w[i][j]
		print("--- END EPOCH {} ----").format(epoch)

		
	#return v, w

	#print("v = {}").format(v)
	#print("w = {}").format(w)

	return x_axis, y_axis


def backpropagation_dataset(v, w, dataset, y_axis):
	"""Calculates the gradients over the entire training set.
	"""

	# get all the y-values from the dataset as a vector
	y = []
	for i in range(len(dataset)):
		y.append(dataset[i][2])

	y_hat_all = forward_computation_dataset(w, v, dataset) # compute prediction for every input
	y_axis.append(loss(y, y_hat_all))

	#print("total loss: {}").format(loss(y, y_hat_all))

	# calculate the derivative loss for the backpropagation
	loss_der_now = loss_der(y, y_hat_all)
	print("LOSS DER : {}").format(loss_der_now)

	print("using y = {} and y_hat = {}").format(y, y_hat_all)
	print("using w = {} and v = {}").format(w, v)

	# contains the gradients over all training elements
	gradient_v_all = init_array(len(v))
	gradient_w_all = init_array(len(w), len(w[0]))

	# call the backpropagation method for every element in the training set
	for i in range(len(dataset)):
		# get the current x from the dataset
		x = list(dataset[i])
		x[len(x)-1] = 1 # override y to be the bias 1

		# we need the activation output for backprop
		h_l1 = forward_computation(w, v, x)[0]

		# compute gradients for this training element
		gr_v, gr_w = backpropagation(v, w, x, h_l1, loss_der_now)

		# add gradient values to the total sum of gradient values
		for j in range(len(gr_v)):
			gradient_v_all[j] += gr_v[j]

		# same for w
		for j in range(len(w)):
			for k in range(len(w[0])):
				gradient_w_all[j][k] += gr_w[j][k]

	# now average the gradient over the number of training examples
	for j in range(len(gr_v)):
		gradient_v_all[j] /= len(dataset)

	for j in range(len(w)):
		for k in range(len(w[0])):
			gradient_w_all[j][k] /= len(dataset)

	print("gradient_w_all: {}").format(gradient_w_all)
	print("gradient_v_all: {}").format(gradient_v_all)

	return gradient_w_all, gradient_v_all

def forward_computation_dataset(w, v, dataset):
	"""Calculates the predictions for the entire training set.
	"""
	print("--- FOWARD COMP ----")

	y_hat_all = []

	for i in range(len(dataset)):
		x = list(dataset[i])
		x[len(x)-1] = 1 # override y to be the bias 1
		y_hat_all.append(forward_computation(w, v, x)[1])

	return y_hat_all


def forward_computation(w, v, x):
	"""Calculates the prediction for a specific input.
	"""

	# first compute the result for the first layer
	a_l1 = init_array(len(w))
	for i in range(len(w)):
		temp = 0
		for j in range(len(w[0])):
			temp += x[j] * w[i][j]
		a_l1[i] = temp

	a_l1.append(1)  # for the bias
	h_l1 = [relu(element) for element in a_l1]

	# then compute the result for the second layer
	y_hat = 0
	for i in range(len(v)):
		y_hat += h_l1[i] * v[i]

	return h_l1, y_hat



def backpropagation(v, w, x, h_l1, loss_der_now, dropped_nodes = []):
	"""Execute backpropgation for one training example.
	Note: Because the result is only the gradient for one training example
	one has to call this method for every training example in the training dataset.
	"""

	print("---- START: BACKPROPAGATION ----")

	# first calculate the gradient for v
	gradient_v = init_array(len(v))
	for i in range(len(v)):
		# ignore all connections to end node from here, if node is dropped
		if dropped_nodes and dropped_nodes[i]:
			continue
		gradient_v[i] = loss_der_now * h_l1[i]
		print("gradient of v_{} = {} * {} = {}").\
		format(i, loss_der_now, h_l1[i], gradient_v[i])

	# then calculate the gradient for w
	gradient_w = init_array(len(w), len(w[0]))
	for i in range(len(w)):
		# ignore all connections to this node if it is dropped
		if dropped_nodes and dropped_nodes[i]:
			continue
		for j in range(len(w[0])):
			gradient_w[i][j] = x[j] * relu_der(h_l1[i]) * v[i] * loss_der_now
			print("gradient of w_{}{} = {} * {} * {} {} = {}").\
			format(i, j, x[j], relu_der(h_l1[i]), v[i], loss_der_now, gradient_w[i][j])

	print("---- END: BACKPROPAGATION ----")



	return gradient_v, gradient_w


def main():

	w, v = xaviar_initalization()
	w2, v2 = constant_initalization(1)
	dataset = [
		[1, 1, 0],
		[-1, 1, 1],
		[1, -1, 1],
		[-1, -1, 0]
	]
	use_dropout = False

	print("----  INPUT ----")
	print("dataset = {}").format(dataset)
	print("w = {}").format(w)
	print("v = {}").format(v)
	print("use dropout = {}\n").format(use_dropout)

	#x_axis2, y_axis2 = rms_prop(w2, v2, dataset, 0.2, 0.5, 50)
	x_axis, y_axis = rms_prop(w, v, dataset, 0.2, 0.5, 50)
	#while y_axis[1] > 5:
	#	x_axis, y_axis = rms_prop(w, v, dataset, 0.2, 0.5, 50)


	#del x_axis[0:2]
	#del y_axis[0:2]

	plt.plot(x_axis, y_axis, 'rx')
	#plt.plot(x_axis2, y_axis2, 'bx')

	plt.ylabel('total loss')
	plt.xlabel('epoch')
	plt.show()

if __name__ == '__main__':
	main()



