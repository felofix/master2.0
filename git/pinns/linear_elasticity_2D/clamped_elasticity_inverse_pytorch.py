import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
import torch.nn.init as init
import pickle
import matplotlib.pyplot as plt

Pi = np.pi

# Parameters
mu = nn.Parameter(torch.tensor(2.5, requires_grad = True))
lambda_ = nn.Parameter(torch.tensor(1.5, requires_grad = True))

def manu_elasticity_inverse_solve(nx, ny, n_hid, n_neu, epochs, lr, activation_function = nn.Tanh(), verbose=False, k = 1, exact_data_type='stress', seed=1234, fixed = None):
	"""
	PARAMETERS:
	n_hid = Number of hidden layers.
	n_neu = Number of neurons in each hidden layer.
	"""
	global mu, lambda_

	torch.manual_seed(seed)
	np.random.seed(seed)

	n_inputs =  2   # x and y.
	n_outputs = 5   # displacement in x and y. 

	# Constants.
	lenght = 1; width = 1
	n_length = nx # + 1
	n_width = ny # + 1

	# Neural network.
	net = Net(n_hid, n_neu, n_inputs, n_outputs, activation_function)
	net = net.to(device)

	if fixed == 'mu':
		mu = 1.0
		optimizer = torch.optim.Adam(list(net.parameters()) + [lambda_] , lr=lr)
	elif fixed == 'lambda':
		lambda_ = 0.5
		optimizer = torch.optim.Adam(list(net.parameters()) + [mu] , lr=lr)

	else:
		optimizer = torch.optim.Adam(list(net.parameters()) + [mu] + [lambda_] , lr=lr)

	x = np.linspace(0, lenght, n_length)
	y = np.linspace(0, width, n_width)
	xij, yij = np.meshgrid(x, y)	
	x_flat, y_flat = xij.flatten(), yij.flatten()
	x_flat_test, y_flat_test = x_flat, y_flat

	# Boundary conditions. 
	left = np.arange(0, n_width * n_length, n_length)
	right = np.arange(n_length - 1, (n_length) * n_width, (n_length))
	top = np.arange(0, n_length) + n_length*(n_length - 1)
	bottom = np.arange(0, n_length) 
	internal = ((x_flat > 0) & (x_flat < lenght) & (y_flat > 0) & (y_flat < width))

	internal_losses = []
	exact_losses = []
	sigma_losses = []
	mus = []
	lambdas = []

	x_flat = np.concatenate((x_flat, x_flat[left], x_flat[top], x_flat[right], x_flat[bottom]), 0)
	y_flat = np.concatenate((y_flat, y_flat[left], y_flat[top], y_flat[right], y_flat[bottom]), 0)

	mu_exac = 1.0
	lambda_exac = 0.5

	exact = u_exact(x_flat, y_flat, k, mu_exac, lambda_exac)
	
	for epoch in range(epochs):
		if exact_data_type == 'stress':
			internal_loss, exact_loss, lame_loss = stress_loss(x_flat, y_flat, net, k, exact)
		elif exact_data_type == 'strain':
			internal_loss, exact_loss, lame_loss = strain_loss(x_flat, y_flat, net, k, exact)
		elif exact_data_type == 'strain_with_boundaries':
			internal_loss, exact_loss, lame_loss = strain_loss_with_boundaries(x_flat, y_flat, net, k, exact, left, right, bottom, top)

		loss = internal_loss + exact_loss + lame_loss

		if fixed == 'mu':
			lambdas.append(lambda_.item())
			mus.append(mu)
		elif fixed == 'lambda':
			lambdas.append(lambda_)
			mus.append(mu.item())

		else:
			lambdas.append(lambda_.item())
			mus.append(mu.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(loss.item(), epoch)

	if not verbose:
		return net, mu.item(), lambda_.item()
	else:
		return mus, lambdas

class Net(nn.Module):
	def __init__(self, num_hidden_layers, num_neurons, ninputs, noutputs, activation_function):
		#Initializing the neural network. 
		#Trying to learn how the network runs. 
		
		super(Net, self).__init__()
		self.num_hidden_layers = num_hidden_layers
		self.num_neurons = num_neurons
		self.ninputs = ninputs
		self.noutputs = noutputs
		self.hidden_layers = nn.ModuleList()
		self.hidden_layers.append(nn.Linear(self.ninputs, self.num_neurons))
		self.activation_function = activation_function

		for hl in range(1, self.num_hidden_layers - 1):
			self.hidden_layers.append(nn.Linear(self.num_neurons, self.num_neurons))

		self.output_layer = nn.Linear(self.num_neurons, self.noutputs)

	def forward(self, inputs):  
		#Moving the neural network forward. 
		#Forward step of the neural network. 
		
		layer_inputs = torch.cat(inputs, dim=1) 

		layer = self.activation_function(self.hidden_layers[0](layer_inputs))

		for hl in range(1, self.num_hidden_layers - 1):
			layer = self.activation_function(self.hidden_layers[hl](layer))

		output = self.output_layer(layer) 
		return output

def stress_loss(x, y, net, k, exact):
	x = create_tensor(x.reshape(-1, 1))
	y = create_tensor(y.reshape(-1, 1))

	# Forces.
	f_x = 2*Pi*Pi*k*k*mu*torch.sin(Pi*k*x) + Pi*Pi*k*k*mu*torch.sin(Pi*k*y)*torch.cos(Pi*k*x) - \
		  lambda_*(-Pi*Pi*k*k*torch.sin(Pi*k*x) - Pi*Pi*k*k*torch.sin(Pi*k*y)*torch.cos(Pi*k*x))

	f_y = Pi*Pi*k*k*lambda_*torch.sin(Pi*k*x)*torch.cos(Pi*k*y) + 3*Pi*Pi*k*k*mu*torch.sin(Pi*k*x)*torch.cos(Pi*k*y)

	# Tensors.
	f_x = create_tensor(f_x.reshape((-1, 1)))
	f_y = create_tensor(f_y.reshape((-1, 1)))

	# Predicted values. 
	u = net([x, y])
	u_x = u[:, 0].reshape((-1, 1))
	u_y = u[:, 1].reshape((-1, 1))
	s_xx = u[:, 2].reshape((-1, 1))
	s_yy = u[:, 3].reshape((-1, 1))
	s_xy = u[:, 4].reshape((-1, 1))

	epsilon_xx, epsilon_yy, epsilon_xy, sigma_xx, sigma_yy, sigma_xy = create_sigma(u_x, u_y, x, y)

	# Sigma losses.
	sxx_loss = torch.mean(torch.square(sigma_xx - s_xx))
	syy_loss = torch.mean(torch.square(sigma_yy - s_yy))
	sxy_loss = torch.mean(torch.square(sigma_xy - s_xy))
	sigma_loss = sxx_loss + syy_loss + sxy_loss

	# Internal losses. 
	sigma_x = diff(s_xx, x) + diff(s_xy, y) + f_x
	sigma_y = diff(s_xy, x) + diff(s_yy, y) + f_y
	sigma_x_loss = torch.mean(torch.square(sigma_x))
	sigma_y_loss = torch.mean(torch.square(sigma_y)) 
	internal_loss = sigma_x_loss + sigma_y_loss

	# Exact loss.
	u_x_loss = torch.mean(torch.square(u_x - exact['u_x']))
	u_y_loss = torch.mean(torch.square(u_y - exact['u_y']))

	exact_sxx_loss = torch.mean(torch.square(s_xx - exact['s_xx']))
	exact_syy_loss = torch.mean(torch.square(s_yy - exact['s_yy']))
	exact_sxy_loss = torch.mean(torch.square(s_xy - exact['s_xy']))

	exact_loss = u_x_loss + u_y_loss + exact_sxx_loss + exact_syy_loss + exact_sxy_loss

	return internal_loss, exact_loss, sigma_loss

def strain_loss(x, y, net, k, exact):
	x = create_tensor(x.reshape(-1, 1))
	y = create_tensor(y.reshape(-1, 1))

	# Forces.
	f_x = 2*Pi*Pi*k*k*mu*torch.sin(Pi*k*x) + Pi*Pi*k*k*mu*torch.sin(Pi*k*y)*torch.cos(Pi*k*x) - \
		  lambda_*(-Pi*Pi*k*k*torch.sin(Pi*k*x) - Pi*Pi*k*k*torch.sin(Pi*k*y)*torch.cos(Pi*k*x))

	f_y = Pi*Pi*k*k*lambda_*torch.sin(Pi*k*x)*torch.cos(Pi*k*y) + 3*Pi*Pi*k*k*mu*torch.sin(Pi*k*x)*torch.cos(Pi*k*y)

	# Tensors.
	f_x = create_tensor(f_x.reshape((-1, 1)))
	f_y = create_tensor(f_y.reshape((-1, 1)))

	# Predicted values. 
	u = net([x, y])
	u_x = u[:, 0].reshape((-1, 1))
	u_y = u[:, 1].reshape((-1, 1))
	e_xx = u[:, 2].reshape((-1, 1))
	e_yy = u[:, 3].reshape((-1, 1))
	e_xy = u[:, 4].reshape((-1, 1))

	epsilon_xx, epsilon_yy, epsilon_xy, sigma_xx, sigma_yy, sigma_xy = create_sigma(u_x, u_y, x, y)

	# epsilon losses.
	exx_loss = torch.mean(torch.square(epsilon_xx - e_xx))
	eyy_loss = torch.mean(torch.square(epsilon_yy - e_yy))
	exy_loss = torch.mean(torch.square(epsilon_xy - e_xy))
	epsilon_loss = exx_loss + eyy_loss + exy_loss
	
	# Internal losses. 
	sigma_x = diff(sigma_xx, x) + diff(sigma_xy, y) + f_x
	sigma_y = diff(sigma_xy, x) + diff(sigma_yy, y) + f_y
	sigma_x_loss = torch.mean(torch.square(sigma_x))
	sigma_y_loss = torch.mean(torch.square(sigma_y)) 
	internal_loss = sigma_x_loss + sigma_y_loss

	# Exact loss.
	e_x_loss = torch.mean(torch.square(u_x - exact["u_x"]))
	e_y_loss = torch.mean(torch.square(u_y - exact["u_y"]))

	exact_exx_loss = torch.mean(torch.square(e_xx - exact["e_xx"])) 
	exact_eyy_loss = torch.mean(torch.square(e_yy - exact["e_yy"]))
	exact_exy_loss = torch.mean(torch.square(e_xy - exact["e_xy"]))

	exact_loss = e_x_loss + e_y_loss + exact_exx_loss + exact_eyy_loss + exact_exy_loss

	return internal_loss, exact_loss, epsilon_loss

def create_sigma(u_x, u_y, x, y):
	du_x_x = diff(u_x, x)
	du_y_y = diff(u_y, y)
	du_x_y = diff(u_x, y)
	du_y_x = diff(u_y, x)

	e_xx = du_x_x
	e_yy = du_y_y
	e_xy = (du_x_y + du_y_x)/2.0
	e_kk = e_xx + e_yy

	sigma_xx = lambda_*e_kk + 2*mu*e_xx
	sigma_yy = lambda_*e_kk + 2*mu*e_yy
	sigma_xy = 2*mu*e_xy 

	return e_xx, e_yy, e_xy, sigma_xx, sigma_yy, sigma_xy

def create_tensor(k):
	tensor = torch.tensor(k, dtype=torch.float32, requires_grad=True).to(device)
	return tensor

def diff(u, d):
	return torch.autograd.grad(u, d, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

def predict(network, xij, yij):
	xij = xij.reshape((len(xij.flatten()), 1))
	yij = yij.reshape((len(yij.flatten()), 1))

	tx = create_tensor(xij)
	ty = create_tensor(yij)
	deform = network([tx, ty]).detach().numpy()

	return deform

def u_exact(x, y, k, mu, lambd):
	# Constans. 
	pi = np.pi
	cos = np.cos
	sin = np.sin

	u_x = create_tensor((np.sin(k*np.pi*x)).reshape(-1, 1))
	u_y = create_tensor((np.sin(k*np.pi*x)*np.cos(k*np.pi*y)).reshape(-1, 1))

	e_xx = create_tensor((pi*k*cos(pi*k*x)).reshape(-1, 1))
	e_yy = create_tensor((-pi*k*sin(pi*k*x)*sin(pi*k*y)).reshape(-1, 1))
	e_xy = create_tensor((0.5*pi*k*cos(pi*k*x)*cos(pi*k*y)).reshape(-1, 1))

	s_xx = create_tensor((2*pi*k*mu*cos(pi*k*x) + lambd*(-pi*k*sin(pi*k*x)*sin(pi*k*y) + pi*k*cos(pi*k*x))).reshape(-1, 1))
	s_yy = create_tensor((-2*pi*k*mu*sin(pi*k*x)*sin(pi*k*y) + lambd*(-pi*k*sin(pi*k*x)*sin(pi*k*y) + pi*k*cos(pi*k*x))).reshape(-1, 1))
	s_xy = create_tensor((1.0*pi*k*mu*cos(pi*k*x)*cos(pi*k*y)).reshape(-1, 1))

	exact = {"u_x": u_x, "u_y": u_y, "e_xx": e_xx, "e_yy": e_yy, "e_xy": e_xy, "s_xx": s_xx, "s_yy": s_yy, "s_xy": s_xy}

	
	return exact













