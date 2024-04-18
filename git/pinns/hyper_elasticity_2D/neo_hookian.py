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

class Neo_hookian:
	def __init__(self, nx, ny, n_hid, n_neu, epochs, problem = 'forward', \
				 lr=1e-3, activation_function = nn.Tanh(),\
				 verbose=False, n_inputs = 2, n_outputs = 6, exact = None, seed=1234, fixed='None'):

		torch.manual_seed(seed)
		np.random.seed(seed)

		# Model specifics. 
		self.problem = problem
		self.exact = exact
		self.n_outputs = n_outputs

		# Neural network.
		self.net = Net(n_hid, n_neu, n_inputs, n_outputs, activation_function)
		self.net = self.net.to(device)
		self.epochs = epochs

		if self.problem == "forward":
			self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
			self.mu = 1
			self.kappa = 0.5

		elif self.problem == "inverse":
			if fixed == 'mu':
				self.mu = create_tensor(1.0)
				self.kappa = nn.Parameter(torch.tensor(2.0, requires_grad = True))
				self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.kappa], lr=lr)

			elif fixed == 'kappa':
				self.mu = nn.Parameter(torch.tensor(5.0, requires_grad = True))
				self.kappa = create_tensor(0.5)
				self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.mu], lr=lr)

			else:
				self.mu = nn.Parameter(torch.tensor(5.0, requires_grad = True))
				self.kappa = nn.Parameter(torch.tensor(2.0, requires_grad = True))
				self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.mu] + [self.kappa], lr=lr)

		# Dimensions
		self.nx = nx
		self.ny = ny 

		self.nxL = 1
		self.nyL = 1

		# Doing things.
		if problem == 'forward':
			self.create_domain()
			self.create_boundaries()
		else:
			self.create_domain(X = exact["x"], Y = exact["y"])
			self.create_boundaries()

		# lists of things.
		self.losses = []
		self.mus = []
		self.kappas = []

	def create_boundaries(self):
		"""
		As for now these are just for a 2D rectangle problem. 
		"""

		# Boundary conditions. 
		self.left = np.where(self.X == 0)
		self.right = np.where(self.X == 1)
		self.top = np.where(self.Y == 1)
		self.bottom = np.where(self.Y == 0)

	def create_domain(self, X = [], Y = []):
		"""
		For now a 2d rectnagle problem. 
		"""
		if self.problem == "forward":
			X = np.linspace(0, self.nxL, self.nx)
			Y = np.linspace(0, self.nyL, self.ny)
			Xij, Yij = np.meshgrid(X, Y)	
			X, Y = Xij.flatten(), Yij.flatten()
			self.X, self.Y = X, Y
			self.X_tensor = create_tensor(X.reshape(-1, 1))
			self.Y_tensor = create_tensor(Y.reshape(-1, 1))
		else: 
			self.X, self.Y = X, Y

	def solve(self):
		"""
		Solving specific problem. 
		"""

		for self.epoch in range(self.epochs):
			if self.problem == 'forward':
				loss = self.forward_loss()
			elif self.problem == 'inverse':
				loss = self.inverse_loss()
				self.mus.append(self.mu.item())
				self.kappas.append(self.kappa.item())

			print(loss.item(), self.epoch)

			# For displacement formulation.
			if self.n_outputs == 2 and self.epoch == self.epochs - 1:
				X = np.linspace(0, self.nxL, 100) # For the beauty print.
				Y = np.linspace(0, self.nyL, 100)
				Xij, Yij = np.meshgrid(X, Y)	
				X, Y = Xij.flatten(), Yij.flatten()
				X, Y, = create_tensor(X.reshape(-1, 1)), create_tensor(Y.reshape(-1, 1))
				u = self.net([X, Y])

				self.PK_xx, self.PK_xy, self.PK_yx, self.PK_yy = self.piola_kirchhoff(u[:, 0], u[:, 1], X, Y) 

			self.losses.append(loss.item())
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()


	def forward_loss(self):
		"""
		can be changed, made now for just one problem. 
		"""
		L, B, R, T = self.left, self.bottom, self.right, self.top
		
		# Create tensors.
		X = self.X_tensor
		Y = self.Y_tensor

		if self.n_outputs == 6:
			# Predicted values. 
			u = self.net([X, Y])
			u_X = u[:, 0].reshape((-1, 1))
			u_Y = u[:, 1].reshape((-1, 1))
			P_xx = u[:, 2].reshape((-1, 1))
			P_yy = u[:, 3].reshape((-1, 1))
			P_xy = u[:, 4].reshape((-1, 1))
			P_yx = u[:, 5].reshape((-1, 1))
		else:
			u = self.net([X, Y])
			u_X = u[:, 0].reshape((-1, 1))
			u_Y = u[:, 1].reshape((-1, 1))

		# Calculated Piola - Kirchhoff stresses.
		PK_xx, PK_xy, PK_yx, PK_yy = self.piola_kirchhoff(u_X, u_Y, X, Y) 

		if self.n_outputs == 6:
			# Piola losses.
			pxx_loss = torch.mean(torch.square(PK_xx - P_xx))
			pyy_loss = torch.mean(torch.square(PK_yy - P_yy))
			pxy_loss = torch.mean(torch.square(PK_xy - P_xy))
			pyx_loss = torch.mean(torch.square(PK_yx - P_yx))
			PK_loss = pxx_loss + pyy_loss + pxy_loss + pyx_loss
		else:
			PK_loss = 0
			P_xx = PK_xx
			P_xy = PK_xy
			P_yx = PK_yx
			P_yy = PK_yy

		# Internal losses. 
		div_x = diff(P_xx, X) + diff(P_xy, Y) 
		div_y = diff(P_yx, X) + diff(P_yy, Y) 
		div_x_loss = torch.mean(torch.square(div_x))
		div_y_loss = torch.mean(torch.square(div_y)) 
		internal_loss = div_x_loss + div_y_loss

		# Dirichlet loss. 
		d_X_prep = torch.cat([u_X[L]], dim=0)
		d_Y_prep = torch.cat([u_Y[L]], dim=0)
		dirichlet_loss = torch.mean(torch.square(d_X_prep)) + torch.mean(torch.square(d_Y_prep))

		# Neumann loss. Gotta be a way to shortn this down. 
		t_top_loss_x = P_xy[T] 
		t_top_loss_y = P_yy[T] + 0.3
		t_bottom_loss_x = P_xy[B]
		t_bottom_loss_y = P_yy[B]
		t_right_loss_x = P_xx[R]
		t_right_loss_y = P_yx[R]

		t_x_prep = torch.cat([t_top_loss_x, t_bottom_loss_x, t_right_loss_x], dim=0)
		t_y_prep = torch.cat([t_top_loss_y, t_bottom_loss_y, t_right_loss_y], dim=0)

		neumann_loss = torch.mean(torch.square(t_x_prep)) + torch.mean(torch.square(t_y_prep))

		total_loss = PK_loss + internal_loss + dirichlet_loss*2 + neumann_loss*1e1

		return total_loss

	def inverse_loss(self):
		"""
		can be changed, made now for just one problem. 
		"""
		X, Y = self.X, self.Y
		
		# Create tensors.
		X = create_tensor(X.reshape(-1, 1))
		Y = create_tensor(Y.reshape(-1, 1))

		# Predicted values. 
		u = self.net([X, Y])
		u_X = u[:, 0].reshape((-1, 1))
		u_Y = u[:, 1].reshape((-1, 1))
		P_xx = u[:, 2].reshape((-1, 1))
		P_yy = u[:, 3].reshape((-1, 1))
		P_xy = u[:, 4].reshape((-1, 1))
		P_yx = u[:, 5].reshape((-1, 1))

		# Calculated Piola - Kirchhoff stresses.
		PK_xx, PK_xy, PK_yx, PK_yy = self.piola_kirchhoff(u_X, u_Y, X, Y) 

		# Piola loss.
		pxx_loss = torch.mean(torch.square(PK_xx - P_xx))
		pyy_loss = torch.mean(torch.square(PK_yy - P_yy))
		pxy_loss = torch.mean(torch.square(PK_xy - P_xy))
		pyx_loss = torch.mean(torch.square(PK_yx - P_yx))
		PK_loss = pxx_loss + pyy_loss + pxy_loss + pyx_loss

		# Internal loss. 
		div_x = diff(P_xx, X) + diff(P_xy, Y) 
		div_y = diff(P_yx, X) + diff(P_yy, Y) 
		div_x_loss = torch.mean(torch.square(div_x))
		div_y_loss = torch.mean(torch.square(div_y)) 
		internal_loss = div_x_loss + div_y_loss

		# Exact loss. 
		u_x_loss = torch.mean(torch.square(u_X - create_tensor(self.exact['u_x'].reshape(-1, 1))))
		u_y_loss = torch.mean(torch.square(u_Y - create_tensor(self.exact['u_y'].reshape(-1, 1))))

		exact_pxx_loss = torch.mean(torch.square(P_xx - create_tensor(self.exact['P_xx'].reshape(-1, 1))))
		exact_pyy_loss = torch.mean(torch.square(P_yy - create_tensor(self.exact['P_yy'].reshape(-1, 1))))
		exact_pxy_loss = torch.mean(torch.square(P_xy - create_tensor(self.exact['P_xy'].reshape(-1, 1))))
		exact_pyx_loss = torch.mean(torch.square(P_yx - create_tensor(self.exact['P_yx'].reshape(-1, 1))))

		exact_loss = u_x_loss + u_y_loss + exact_pxx_loss + exact_pyy_loss + exact_pxy_loss + exact_pyx_loss

		total_loss = exact_loss + PK_loss + internal_loss

		return total_loss

	def piola_kirchhoff(self, ux, uy, X, Y):
		"""
		Calculates the deformation gradient using autograd. Changed grad too diff 
		when I should actually create the function. 
		"""

		ux_x = diff(ux, X) + 1
		ux_y = diff(ux, Y)

		uy_x = diff(uy, X)
		uy_y = diff(uy, Y) + 1

		F = [[ux_x, ux_y], 
			 [uy_x, uy_y]]

		J = F[0][0]*F[1][1] - F[0][1]*F[1][0]

		Finv =  [[uy_y/J, -ux_y/J], 
			 	 [-uy_x/J,  ux_x/J]]

		FTinv = [[Finv[0][0], Finv[1][0]], 
			 	 [Finv[0][1], Finv[1][1]]]

		PK_xx = self.mu*F[0][0] - self.mu*FTinv[0][0] + self.kappa*torch.log(J)*FTinv[0][0]
		PK_yy = self.mu*F[1][1] - self.mu*FTinv[1][1] + self.kappa*torch.log(J)*FTinv[1][1]
		PK_xy = self.mu*F[0][1] - self.mu*FTinv[0][1] + self.kappa*torch.log(J)*FTinv[0][1]
		PK_yx = self.mu*F[1][0] - self.mu*FTinv[1][0] + self.kappa*torch.log(J)*FTinv[1][0]

		return PK_xx, PK_xy, PK_yx, PK_yy

	def predict(self, xij, yij):
		xij = xij.reshape((len(xij.flatten()), 1))
		yij = yij.reshape((len(yij.flatten()), 1))
		tx = create_tensor(xij)
		ty = create_tensor(yij)

		deform = self.net([tx, ty])

		u = deform.detach().numpy()

		if self.n_outputs == 2:
			PK_xx = self.PK_xx.detach().numpy()
			PK_xy = self.PK_xy.detach().numpy()
			PK_yx = self.PK_yx.detach().numpy()
			PK_yy = self.PK_yy.detach().numpy()

			u = np.hstack((u, PK_xx, PK_xy, PK_yx, PK_yy))

		return u

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

def create_tensor(k):
	tensor = torch.tensor(k, dtype=torch.float32, requires_grad=True).to(device)
	return tensor

def diff(u, d):
	return torch.autograd.grad(u, d, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
