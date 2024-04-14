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

torch.manual_seed(1234)
np.random.seed(1234)

mu = 1
lambda_ = 0.5
Pi = np.pi


def manu_elasticity_solve(nx, ny, n_hid, n_neu, epochs, lr, activation_function = nn.Tanh(), verbose=False, k = 1):
	"""
	PARAMETERS:
	n_hid = Number of hidden layers.
	n_neu = Number of neurons in each hidden layer.
	"""

	n_inputs =  2   # x and y.
	n_outputs = 2   # displacement in x and y. 

	# Constants.
	lenght = 1; width = 1
	n_length = nx # + 1
	n_width = ny # + 1

	# Neural network.
	net = Net(n_hid, n_neu, n_inputs, n_outputs, activation_function)
	net = net.to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)

	x = np.linspace(0, lenght, n_length)
	y = np.linspace(0, width, n_width)
	xij, yij = np.meshgrid(x, y)	
	x_flat, y_flat = xij.flatten(), yij.flatten()

	prettyx, prettyy = np.linspace(0, 1, 200), np.linspace(0, 1, 200)
	prettyx, prettyy = np.meshgrid(prettyx, prettyy)	
	prettyx, prettyy = prettyx.flatten(), prettyy.flatten()	

	# Boundary conditions. 
	left = np.arange(0, n_width * n_length, n_length)
	right = np.arange(n_length - 1, (n_length) * n_width, (n_length))
	top = np.arange(0, n_length) + n_length*(n_length - 1)
	bottom = np.arange(0, n_length) 
	internal = ((x_flat > 0) & (x_flat < lenght) & (y_flat > 0) & (y_flat < width))

	internal_losses = []
	dirichlet_losses = []
	neumann_losses = []

	f_x = 2*Pi*Pi*k*k*mu*np.sin(Pi*k*x_flat) + Pi*Pi*k*k*mu*np.sin(Pi*k*y_flat)*np.cos(Pi*k*x_flat) - \
		  lambda_*(-Pi*Pi*k*k*np.sin(Pi*k*x_flat) - Pi*Pi*k*k*np.sin(Pi*k*y_flat)*np.cos(Pi*k*x_flat))

	f_y = Pi*Pi*k*k*lambda_*np.sin(Pi*k*x_flat)*np.cos(Pi*k*y_flat) + 3*Pi*Pi*k*k*mu*np.sin(Pi*k*x_flat)*np.cos(Pi*k*y_flat)

	sigmas = 0
	
	for epoch in range(epochs):
		internal_loss, dirichlet_loss, neumann_loss = loss_function(x_flat, y_flat, net, left, right, bottom, top, internal, f_x, f_y, epoch, k)
		
		internal_losses.append(internal_loss.item())
		dirichlet_losses.append(dirichlet_loss.item())
		neumann_losses.append(neumann_loss.item())

		loss = internal_loss + dirichlet_loss + neumann_loss

		print(dirichlet_loss.item(), epoch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if not verbose:
		return net
	else:
		prettyx, prettyy = create_tensor(prettyx.reshape(-1, 1)), create_tensor(prettyy.reshape(-1, 1))
		u_pretty = net([prettyx, prettyy])
		sigma_xx, sigma_yy, sigma_xy = create_sigma(u_pretty[:, 0], u_pretty[:, 1], prettyx, prettyy)
		return net, internal_losses, dirichlet_losses, neumann_losses, [sigma_xx.detach().numpy(), sigma_yy.detach().numpy(), sigma_xy.detach().numpy()]

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

def loss_function(x, y, net, left, right, bottom, top, internal, f_x, f_y, epoch, k):
	# Tractions. 
	t_x_bottom = create_tensor(-mu*Pi*k*np.cos(k*Pi*x[bottom])*np.cos(k*Pi*y[bottom])).reshape((-1, 1))
	t_y_bottom =  create_tensor(-((2*Pi*k*mu*np.sin(Pi*k*x[bottom])*np.sin(Pi*k*y[bottom]) + \
							lambda_*(-Pi*k*np.sin(Pi*k*x[bottom])*np.sin(Pi*k*y[bottom]) + Pi*k*np.cos(Pi*k*x[bottom])))).reshape((-1, 1)))

	t_x_top = create_tensor(mu*Pi*k*np.cos(k*Pi*x[top])*np.cos(k*Pi*y[top])).reshape((-1, 1))
	t_y_top = create_tensor((2*Pi*k*mu*np.sin(Pi*k*x[top])*np.sin(Pi*k*y[top]) + \
							lambda_*(-Pi*k*np.sin(Pi*k*x[top])*np.sin(Pi*k*y[top]) + Pi*k*np.cos(Pi*k*x[top]))).reshape((-1, 1)))

	# Normals.
	bottom_x = 0
	bottom_y = -1

	top_x = 0 
	top_y = 1

	f_x = create_tensor(f_x.reshape((-1, 1)))
	f_y = create_tensor(f_y.reshape((-1, 1)))

	x = create_tensor(x.reshape(-1, 1))
	y = create_tensor(y.reshape(-1, 1))

	# Predicted values. 
	u = net([x, y])
	u_x = u[:, 0].reshape((-1, 1))
	u_y = u[:, 1].reshape((-1, 1))

	sigma_xx, sigma_yy, sigma_xy = create_sigma(u_x, u_y, x, y)

	# Internal losses. 
	sigma_x = diff(sigma_xx, x) + diff(sigma_xy, y) + f_x
	sigma_y = diff(sigma_xy, x) + diff(sigma_yy, y) + f_y
	sigma_x_loss = torch.mean(torch.square(sigma_x))
	sigma_y_loss = torch.mean(torch.square(sigma_y)) 
	internal_loss = sigma_x_loss + sigma_y_loss

	# Dirichlet loss.
	d_left_x_prep = torch.cat([u_x[left], u_x[right]], dim=0)
	d_left_y_prep = torch.cat([u_y[left], u_y[right]], dim=0)

	dirichlet_loss = torch.mean(torch.square(d_left_x_prep)) + torch.mean(torch.square(d_left_y_prep))
	
	# Neumann loss. 
	# Leftside. 
	t_x_bottom_pred = sigma_xx[bottom]*bottom_x + sigma_xy[bottom]*bottom_y
	t_y_bottom_pred = sigma_xy[bottom]*bottom_x + sigma_yy[bottom]*bottom_y
	t_bottom_x_l = t_x_bottom_pred - t_x_bottom
	t_bottom_y_l = t_y_bottom_pred - t_y_bottom

	# Top.
	t_x_top_pred = sigma_xx[top]*top_x + sigma_xy[top]*top_y
	t_y_top_pred = sigma_xy[top]*top_x + sigma_yy[top]*top_y
	t_top_x_l = t_x_top_pred - t_x_top
	t_top_y_l = t_y_top_pred - t_y_top
	
	t_left_x_prep = torch.cat([t_bottom_x_l, t_top_x_l], dim=0)
	t_left_y_prep = torch.cat([t_bottom_y_l, t_top_y_l], dim=0)
	
	neumann_loss = torch.mean(torch.square(t_left_x_prep)) + torch.mean(torch.square(t_left_y_prep))

	return internal_loss, dirichlet_loss, neumann_loss

def create_tensor(k):
	tensor = torch.tensor(k, dtype=torch.float32, requires_grad=True).to(device)
	return tensor

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

	return sigma_xx, sigma_yy, sigma_xy

def diff(u, d):
	return torch.autograd.grad(u, d, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

def predict(network, xij, yij):
	xij = xij.reshape((len(xij.flatten()), 1))
	yij = yij.reshape((len(yij.flatten()), 1))

	tx = create_tensor(xij)
	ty = create_tensor(yij)
	deform = network([tx, ty]).detach().numpy()

	return deform














