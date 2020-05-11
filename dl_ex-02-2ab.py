import random

def relu(value): return 0 if value < 0 else value

def relu_der(value): return 0 if value <= 0 else 1 

def loss(y, y_hat):	return (y - y_hat)**2

def loss_der(y, y_hat):	return -2 * (y - y_hat)

def init_array(i, j = 0):
	if j is 0:
		return [0 for k in range(i)]
	else:
		return [[0 for m in range(j)] for k in range(i)]

def nodes_to_drop_out(layer_size):
	return [random.uniform(0, 1) > 0.8 for i in range(layer_size)]

def stochastic_gd(w, v, gradient_v, gradient_w, eta):

	print("---- START: SGD ----")

	# update the v values
	v_new = init_array(len(v))
	for i in range(len(v)):
		v_new[i] = v[i] - eta * gradient_v[i]
		print("v_new_{} = {} - {} * {} = {}")\
		.format(i, v[i], eta, gradient_v[i], v_new[i])

	# update the w values
	w_new = init_array(len(w), len(w[0]))
	for i in range(len(w)):
		for j in range(len(w[0])):
			w_new[i][j] = w[i][j] - eta * gradient_w[i][j]
			print("w_new_{}{} = {} - {} * {} = {}")\
			.format(i, j, w[i][j], eta, gradient_w[i][j], w_new[i][j])

	print("---- END: SGD ----")

	return v_new, w_new

def forward_computation(w, v, x):

	print("---- START: FORWARD COMPUTATION ----")

	# first compute the result for the first layer
	a_l1 = init_array(len(w))
	for i in range(len(w)):
		temp = 0
		for j in range(len(w[0])):
			temp += x[j] * w[i][j]
		a_l1[i] = temp

	a_l1.append(1)  # for the bias
	h_l1 = [relu(element) for element in a_l1]

	print("a_layer_1 = {}").format(a_l1)
	print("h_layer_1 = {}").format(h_l1)

	# then compute the result for the second layer
	y_hat = 0
	for i in range(len(v)):
		y_hat += h_l1[i] * v[i]

	print("y_hat = {}").format(y_hat)

	print("---- END: FORWARD COMPUTATION ----")

	return h_l1, y_hat


def backpropagation(x, v, w, h_l1, y_hat, y, dropped_nodes):

	print("---- START: BACKPROPAGATION ----")

	# first calculate the gradient for v
	gradient_v = init_array(len(v))
	for i in range(len(v)):
		# ignore all connections to end node from here, if node is dropped
		if dropped_nodes and dropped_nodes[i]:
			print("gradient of v_{} = 0 (dropped)").format(i)
			continue
		gradient_v[i] = loss_der(y, y_hat) * h_l1[i]
		print("gradient of v_{} = {} * {} = {}").\
		format(i, loss_der(y, y_hat), h_l1[i], gradient_v[i])

	# then calculate the gradient for w
	gradient_w = init_array(len(w), len(w[0]))
	for i in range(len(w)):
		# ignore all connections to this node if it is dropped
		if dropped_nodes and dropped_nodes[i]:
			print("gradient of w_{}j = 0 for j = 0,...,{} (dropped)").format(i, len(w[0])-1)
			continue
		for j in range(len(w[0])):
			gradient_w[i][j] = x[j] * relu_der(h_l1[i]) * v[i] * loss_der(y, y_hat)
			print("gradient of w_{}{} = {} * {} * {} {} = {}").\
			format(i, j, x[j], relu_der(h_l1[i]), v[i], loss_der(y, y_hat), gradient_w[i][j])

	print("---- END: BACKPROPAGATION ----")

	return gradient_v, gradient_w


def iteration_step(w, v, x, y, eta, use_dropout, dropped_nodes = []):

	# start by computating the activation function output for layer 1
	# and the output y_hat at the output node
	h_l1, y_hat = forward_computation(w, v, x)

	loss_val = loss(y, y_hat)

	if use_dropout:

		if not dropped_nodes:
			# if the user did not provide dropped nodes, drop randomly
			dropped_nodes = nodes_to_drop_out(len(w))
		
		print("dropped nodes: {}").format(dropped_nodes)
		dropped_nodes.append(False) # one more for the second-layer-bias which is never dropped

	gradient_v, gradient_w = backpropagation(x, v, w, h_l1, y_hat, y, dropped_nodes)

	v_new, w_new = stochastic_gd(w, v, gradient_v, gradient_w, eta)

	return v_new, w_new, y_hat

def main():

	w = [[-1, -2, 1], [-3, 2, 2], [1, 3, 1]]
	v = [1, -2, -1, 1]
	x = [3, 2, 1]
	y = 2
	eta = 0.01
	use_dropout = True

	# provide a node dropout list, instead of randomly dropping nodes
	# only used for testing purposes, True = drop, False = dont drop
	# dropped_nodes_all = [[False, False, False], [False, True, False], [True, False, False]]
	dropped_nodes_all = []

	# do task 2.b
	# start with default w and v
	w_new = w
	v_new = v
	y_hat = 0
	for i in range(3):
		print("ITERATION {}: ").format(i+1)

		if not dropped_nodes_all:
			v_new, w_new, y_hat = iteration_step(w_new, v_new, x, y, eta, use_dropout)
		else:
			v_new, w_new, y_hat = iteration_step(w_new, v_new, x, y, eta, use_dropout, dropped_nodes_all[i])

		print("w_new: {}").format(v_new)
		print("v_new: {}").format(w_new)

		y_hat = forward_computation(w_new, v_new, x)[1]
		print("loss with new w, v: {}").format(loss(y, y_hat))
		print("END OF ITERATION {}: ").format(i+1)

	# do task 2.a
	x_new = [1,3,1]
	forward_computation(w_new, v_new, x_new)[1]


if __name__ == '__main__':
	main()



