import math

def gradient(i, theta):
	result = round(3 * (theta[i])**2 - 2 * theta[i] + 1, 4)
	result_lat = "3 \\cdot {}^2 - 2 \\cdot {} + 1 = {}".format(theta[i], theta[i], result)
	return result, result_lat


def adam_opt(theta, alpha, roh, mu, delta):
	v = [0 for x in range(len(theta))]
	g = [0 for x in range(len(theta))]
	r = [0 for x in range(len(theta))]
	mu_t = [0 for x in range(len(theta))]
	latex_lines = ""

	for i in range(3):

		line_7 = "& g = \\begin{pmatrix}"
		line_8 = "& v = \\begin{pmatrix}"
		line_9 = "& r = \\begin{pmatrix}"
		line_10 = "& \\tilde\\mu = \\begin{pmatrix}"
		line_11 = "& \\theta = \\begin{pmatrix}"

		for j in range(len(theta)):
			g[j], lat = gradient(j, theta)
			line_7 += "{}\\\\".format(lat)
			temp = v[j]
			v[j] = round((1/1-alpha)*(alpha * v[j] + (1-alpha)*g[j]),4)
			line_8 += "\\frac{{1}}{{1-{}}} \\cdot ({} \\cdot {} + (1-{})\\cdot {}) = {} \\\\".format(alpha, alpha, temp, alpha, g[j], v[j])
			temp = r[j]
			r[j] = round((1/1-roh)*(roh * v[j] + (1-roh)*(g[j]**2)),4)
			line_9 += "\\frac{{1}}{{1-{}}} \\cdot ({} \\cdot {} + (1-{})\\cdot {} \\cdot {}) = {} \\\\".format(roh, roh, temp, roh, g[j], g[j], r[j])
			mu_t[j] = round((mu/(roh + math.sqrt(r[j]))),4)
			line_10 += "\\frac{{{}}}{{{}+\\sqrt{{{}}}}} = {} \\\\".format(mu, roh,r[j], mu_t[j])

			temp = theta[j]
			theta[j] = round(theta[j] - mu_t[j] * v[j], 4)
			line_11 += "{} - {} \\cdot {} = {} \\\\".format(temp, mu_t[j], v[j], theta[j])


		line_7 += "\\end{pmatrix} \\\\"
		line_8 += "\\end{pmatrix} \\\\"
		line_9 += "\\end{pmatrix} \\\\"
		line_10 += "\\end{pmatrix} \\\\"
		line_11 += "\\end{pmatrix} \\\\"

		all_lines = "\\begin{{align}} {} {} {} {} {} \\end{{align}}".format(line_7, line_8, line_9, line_10, line_11)
	
		latex_lines += "\n\n ADAM: ITERATION - {}\n\n".format(i)

		latex_lines += all_lines

	return theta, latex_lines


def nestrov_momentum(theta, alpha, mu):
	v = [0 for x in range(len(theta))]
	g = [0 for x in range(len(theta))]
	latex_lines = ""

	for i in range(3):
		line_6 = "& g = \\begin{pmatrix}"
		line_7 = "& v = \\begin{pmatrix}"
		line_8 = "& \\theta = \\begin{pmatrix}"

		for j in range(len(theta)):
			theta_decay = [theta[j] + alpha * v[j]]
			line_6 += "\\nabla f({} + {} \\cdot {}) = ".format(theta[j], alpha, v[j])
			g[j], lat = gradient(0, theta_decay)
			line_6 += "{} \\\\".format(lat)
			temp = v[j]
			v[j] = round(alpha * v[j] - mu * g[j], 4)
			line_7 += "{} \\cdot {} - {} \\cdot {} = {} \\\\".format(alpha, temp, mu, g[j], v[j]) 
			temp = theta[j]
			theta[j] = round(theta[j] + v[j], 4)
			line_8 += "{} + {} = {} \\\\".format(temp, v[j], theta[j])

		line_6 += "\\end{pmatrix} \\\\"
		line_7 += "\\end{pmatrix} \\\\"
		line_8 += "\\end{pmatrix} \\\\"

		all_lines = "\\begin{{align}} {} {} {}\\end{{align}}".format(line_6, line_7, line_8)
		latex_lines += "\n\n NESTROV_MOM: ITERATION - {}\n\n".format(i)
		latex_lines += all_lines

	return theta, latex_lines


def sgd_adagrad(theta, mu, delta):
	r = [0 for x in range(len(theta))]
	g = [0 for x in range(len(theta))]
	mu_t = [0 for x in range(len(theta))]
	latex_lines = ""


	for i in range(3):
		line_6 = "& g = \\begin{pmatrix}"
		line_7 = "& r = \\begin{pmatrix}"
		line_8 = "& \\tilde\\mu = \\begin{pmatrix}"
		line_9 = "& \\theta = \\begin{pmatrix}"
		for j in range(len(theta)):
			g[j], lat = gradient(j, theta)
			line_6 += "{}\\\\".format(lat)
			temp = r[j]
			r[j] = round(r[j] + g[j]**2, 4)
			line_7 += "{} + {}^2 = {}\\\\".format(temp, g[j], r[j])
			mu_t[j] = round( mu / (delta + math.sqrt(r[j])), 4)
			line_8 += "\\frac{{{}}}{{{} + \\sqrt{{{}}}}} = {}\\\\".format(mu, "10^{-8}", r[j], mu_t[j])
			
			temp = theta[j]
			theta[j] = round(theta[j] - mu_t[j] * g[j], 4)
			line_9 += "{} - {} \\cdot {} = {}\\\\".format(temp, mu_t[j], g[j], theta[j])


		line_6 += "\\end{pmatrix} \\\\"
		line_7 += "\\end{pmatrix} \\\\"
		line_8 += "\\end{pmatrix} \\\\"
		line_9 += "\\end{pmatrix} \\\\"

		all_lines = "\\begin{{align}} {} {} {} {} \\end{{align}}".format(line_6, line_7, line_8, line_9)
		latex_lines += "\n\n SGD_ADAGRAD: ITERATION - {}\n\n".format(i)
		latex_lines += all_lines

	return theta, latex_lines

def main():
	theta = [0.1, 0.2, -0.3, -2, 1, -1.5]
	theta_ada, latex_lines_ada = sgd_adagrad(theta, 0.01, 0.00000001)
	theta_mom, latex_lines_mom = nestrov_momentum(theta, 0.5, 0.001)
	theta_adam, latex_lines_adam = adam_opt(theta, 0.9, 0.99, 0.001, 0.00000001)
	print(latex_lines_adam)
	print(latex_lines_mom)
	print(latex_lines_ada)



if __name__ == '__main__':
	main()