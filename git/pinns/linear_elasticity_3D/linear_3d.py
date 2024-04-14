import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
import torch.nn.init as init
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)

Pi = np.pi

class Linear3D:
	def __init__(self, n_hid, n_neu, epochs, problem = 'forward', \
				 lr=1e-3, activation_function = nn.Tanh(),\
				 n_inputs = 3, n_outputs = 9, exact = None, fixed = None):

		# Model specifics. 
		self.problem = problem
		self.exact = exact
		self.fixed = fixed

		# Neural network.
		self.net = Net(n_hid, n_neu, n_inputs, n_outputs, activation_function)
		self.net = self.net.to(device)
		self.epochs = epochs

		if self.problem == "forward":
			self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
			self.mu = 1
			self.lambda_ = 0.5

		elif self.problem == "inverse":
			self.mu = nn.Parameter(torch.tensor(0.5, requires_grad = True))
			self.lambda_ = nn.Parameter(torch.tensor(2.0, requires_grad = True))

			if self.fixed == 'mu':
				self.mu = 1.0
				self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.lambda_], lr=lr)

			elif self.fixed == 'lambda':
				self.lambda_ = 0.5
				self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.mu], lr=lr)

			else:
				self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.mu] + [self.lambda_], lr=lr)
			
		# lists of things.
		self.losses = []
		self.mus = []
		self.lambdas = []

		# Dimensions
		self.nx = 11
		self.ny = 11 
		self.nz = 11

		self.nxL = 1
		self.nyL = 1
		self.nzL = 1

		self.create_domain(exact['x'], exact['y'], exact['z'])
		self.create_normals()
		self.create_boundaries()

	def create_boundaries(self):
		"""
		As for now these are just for a 2D rectangle problem. 
		"""
		nx, ny, nz = self.nx, self.ny, self.nz 

		self.left = np.where(self.X == 0)
		self.right = np.where(self.X == 1)
		self.bottom = np.where(self.Z == 0)
		self.top = np.where(self.Z == 1)
		self.front = np.where(self.Y == 0)
		self.back = np.where(self.Y == 1)

	def create_domain(self, X = [], Y = [], Z = []):
		"""
		For now a 2d rectnagle problem. 
		"""
		if len(X) == 0:
			X = np.linspace(0, self.nxL, self.nx)
			Y = np.linspace(0, self.nyL, self.ny)
			Z = np.linspace(0, self.nzL, self.nz)
			Xij, Yij, Zij = np.meshgrid(X, Y, Z)	
			X, Y, Z = Xij.flatten(), Yij.flatten(), Zij.flatten()
		
		self.X, self.Y, self.Z = X, Y, Z

	def create_normals(self):
		# normals.
		self.left_normal = [-1, 0, 0]
		self.right_normal = [1, 0, 0]
		self.bottom_normal = [0, 0, -1]
		self.top_normal = [0, 0, 1]
		self.front_normal = [0, -1, 0]
		self.back_normal = [0, 1, 0]

	def solve(self):
		"""
		Solving specific problem. 
		"""
		for epoch in range(self.epochs):
			self.epoch = epoch
			if self.problem == 'forward':
				loss = self.forward_loss()
			elif self.problem == 'inverse':
				loss = self.inverse_loss()

				if self.fixed == 'mu':
					self.mus.append(self.mu)
					self.lambdas.append(self.lambda_.item())

				elif self.fixed == 'lambda':
					self.mus.append(self.mu.item())
					self.lambdas.append(self.lambda_)

				else:
					self.mus.append(self.mu.item())
					self.lambdas.append(self.lambda_.item())

			print(loss.item(), epoch)
			self.losses.append(loss.item())
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

	def forward_loss(self):
		X, Y, Z = self.X, self.Y, self.Z
		
		# Create tensors.
		X = create_tensor(X.reshape(-1, 1))
		Y = create_tensor(Y.reshape(-1, 1))
		Z = create_tensor(Z.reshape(-1, 1))

		self.X, self.Y, self.Z = self.X.reshape(-1, 1), self.Y.reshape(-1, 1), self.Z.reshape(-1, 1)

		# Predicted values. 
		u = self.net([X, Y, Z])
		u_X =  u[:, 0].reshape(-1, 1)
		u_Y =  u[:, 1].reshape(-1, 1)
		u_Z =  u[:, 2].reshape(-1, 1)
		s_xx = u[:, 3].reshape(-1, 1)
		s_yy = u[:, 4].reshape(-1, 1)
		s_zz = u[:, 5].reshape(-1, 1)
		s_xy = u[:, 6].reshape(-1, 1)
		s_xz = u[:, 7].reshape(-1, 1)
		s_yz = u[:, 8].reshape(-1, 1)

		# Calculated Piola - Kirchhoff stresses.
		sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz = self.create_sigma(u_X, u_Y, u_Z, X, Y, Z)  

		# Sigma losses.
		sxx_loss = torch.mean(torch.square(sigma_xx - s_xx))
		syy_loss = torch.mean(torch.square(sigma_yy - s_yy))
		szz_loss = torch.mean(torch.square(sigma_zz - s_zz))
		sxy_loss = torch.mean(torch.square(sigma_xy - s_xy))
		sxz_loss = torch.mean(torch.square(sigma_xz - s_xz))
		syz_loss = torch.mean(torch.square(sigma_yz - s_yz))
		sigma_loss = sxx_loss + syy_loss + szz_loss + sxy_loss + sxz_loss + syz_loss

		fz = create_tensor(-0.2)

		# Internal losses. 
		sigma_x = diff(s_xx, X) + diff(s_xy, Y) + diff(s_xz, Z) 
		sigma_y = diff(s_xy, X) + diff(s_yy, Y) + diff(s_yz, Z)
		sigma_z = diff(s_xz, X) + diff(s_yz, Y) + diff(s_zz, Z) + fz
		sigma_x_loss = torch.mean(torch.square(sigma_x))
		sigma_y_loss = torch.mean(torch.square(sigma_y)) 
		sigma_z_loss = torch.mean(torch.square(sigma_z)) 
		internal_loss = sigma_x_loss + sigma_y_loss + sigma_z_loss

		# Boundary losses, should be changes it can taake general boundary conditions. 
		# Dirichlet loss. 
		D_X_prep = torch.cat([u_X[self.left]], dim=0) 
		D_Y_prep = torch.cat([u_Y[self.left]], dim=0) 
		D_Z_prep = torch.cat([u_Z[self.left]], dim=0) 
		dirichlet_loss = torch.mean(torch.square(D_X_prep)) + torch.mean(torch.square(D_Y_prep)) + torch.mean(torch.square(D_Z_prep))
		
		# Neumann loss. 
		tx_front, ty_front, tz_front = self.create_traction(self.front, self.front_normal, 0, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)
		tx_back, ty_back, tz_back = self.create_traction(self.back, self.back_normal, 0, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)
		tx_bottom, ty_bottom, tz_bottom = self.create_traction(self.bottom, self.bottom_normal, 0, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)
		tx_top, ty_top, tz_top = self.create_traction(self.top, self.top_normal, 0, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)
		tx_right, ty_right, tz_right = self.create_traction(self.right, self.right_normal, 0, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)

		t_x_prep = torch.cat([tx_front, tx_back, tx_bottom, tx_top, tx_right], dim=0)
		t_y_prep = torch.cat([ty_front, ty_back, ty_bottom, ty_top, ty_right], dim=0)
		t_z_prep = torch.cat([tz_front, tz_back, tz_bottom, tz_top, tz_right], dim=0)

		neumann_loss = torch.mean(torch.square(t_x_prep)) + torch.mean(torch.square(t_y_prep)) + torch.mean(torch.square(t_z_prep))
		total_loss = dirichlet_loss*2 + sigma_loss + internal_loss + neumann_loss*1e1

		return total_loss

	def inverse_loss(self):
		X, Y, Z = self.X, self.Y, self.Z
		
		# Create tensors.
		X = create_tensor(X.reshape(-1, 1))
		Y = create_tensor(Y.reshape(-1, 1))
		Z = create_tensor(Z.reshape(-1, 1))

		# Predicted values. 
		u = self.net([X, Y, Z])
		u_X =  u[:, 0].reshape(-1, 1)
		u_Y =  u[:, 1].reshape(-1, 1)
		u_Z =  u[:, 2].reshape(-1, 1)
		s_xx = u[:, 3].reshape(-1, 1)
		s_yy = u[:, 4].reshape(-1, 1)
		s_zz = u[:, 5].reshape(-1, 1)
		s_xy = u[:, 6].reshape(-1, 1)
		s_xz = u[:, 7].reshape(-1, 1)
		s_yz = u[:, 8].reshape(-1, 1)

		# Calculated Piola - Kirchhoff stresses.
		sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz = self.create_sigma(u_X, u_Y, u_Z, X, Y, Z)  

		# Sigma losses.
		sxx_loss = torch.mean(torch.square(sigma_xx - s_xx))
		syy_loss = torch.mean(torch.square(sigma_yy - s_yy))
		szz_loss = torch.mean(torch.square(sigma_zz - s_zz))
		sxy_loss = torch.mean(torch.square(sigma_xy - s_xy))
		sxz_loss = torch.mean(torch.square(sigma_xz - s_xz))
		syz_loss = torch.mean(torch.square(sigma_yz - s_yz))
		sigma_loss = sxx_loss + syy_loss + szz_loss + sxy_loss + sxz_loss + syz_loss

		# Internal losses. 
		sigma_x = diff(s_xx, X) + diff(s_xy, Y) + diff(s_xz, Z)
		sigma_y = diff(s_xy, X) + diff(s_yy, Y) + diff(s_yz, Z)
		sigma_z = diff(s_xz, X) + diff(s_yz, Y) + diff(s_zz, Z) + create_tensor(-0.2)
		sigma_x_loss = torch.mean(torch.square(sigma_x))
		sigma_y_loss = torch.mean(torch.square(sigma_y)) 
		sigma_z_loss = torch.mean(torch.square(sigma_z)) 
		internal_loss = sigma_x_loss + sigma_y_loss + sigma_z_loss

		u_x_loss = torch.mean(torch.square(u_X - create_tensor(self.exact['u_x'])))
		u_y_loss = torch.mean(torch.square(u_Y - create_tensor(self.exact['u_y'])))
		u_z_loss = torch.mean(torch.square(u_Z - create_tensor(self.exact['u_z'])))

		exact_sxx_loss = torch.mean(torch.square(s_xx - create_tensor(self.exact['s_xx'])))
		exact_syy_loss = torch.mean(torch.square(s_yy - create_tensor(self.exact['s_yy'])))
		exact_szz_loss = torch.mean(torch.square(s_zz - create_tensor(self.exact['s_zz'])))
		exact_sxy_loss = torch.mean(torch.square(s_xy - create_tensor(self.exact['s_xy'])))
		exact_sxz_loss = torch.mean(torch.square(s_xz - create_tensor(self.exact['s_xz'])))
		exact_syz_loss = torch.mean(torch.square(s_xz - create_tensor(self.exact['s_xz'])))

		exact_loss = u_x_loss + u_y_loss + u_z_loss + exact_sxx_loss + exact_syy_loss + \
				     exact_szz_loss + exact_sxy_loss + exact_sxz_loss + exact_syz_loss
					 

		total_loss = exact_loss + sigma_loss + internal_loss 

		return total_loss
	
	def create_traction(self, bc, normal, t, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz):
		t_x = s_xx[bc]*normal[0] + s_xy[bc]*normal[1] + s_xz[bc]*normal[2]
		t_y = s_xy[bc]*normal[0] + s_yy[bc]*normal[1] + s_yz[bc]*normal[2]
		t_z = s_xz[bc]*normal[0] + s_yz[bc]*normal[1] + s_zz[bc]*normal[2]
		t_x_loss = t_x - t*normal[0]
		t_y_loss = t_y - t*normal[1]
		t_z_loss = t_z - t*normal[2]
		return t_x_loss, t_y_loss, t_z_loss

	def create_sigma(self, u_X, u_Y, u_Z, X, Y, Z):
		"""
		Calculates the deformation gradient using autograd. Changed grad too diff 
		when I should actually create the function. Manual calculations in two dimensions. 
		"""
		# Differentiations.
		du_x_x = diff(u_X, X)
		du_y_y = diff(u_Y, Y)
		du_z_z = diff(u_Z, Z)

		du_x_y = diff(u_X, Y)
		du_y_x = diff(u_Y, X)

		du_x_z = diff(u_X, Z)
		du_z_x = diff(u_Z, X)

		du_y_z = diff(u_Y, Z)
		du_z_y = diff(u_Z, Y)

		# Strains.
		e_xx = du_x_x
		e_yy = du_y_y
		e_zz = du_z_z
		e_xy = (du_x_y + du_y_x)/2.0
		e_xz = (du_x_z + du_z_x)/2.0
		e_yz = (du_y_z + du_z_y)/2.0

		e_kk = e_xx + e_yy + e_zz

		sigma_xx = self.lambda_*e_kk + 2*self.mu*e_xx
		sigma_yy = self.lambda_*e_kk + 2*self.mu*e_yy
		sigma_zz = self.lambda_*e_kk + 2*self.mu*e_zz

		sigma_xy = 2*self.mu*e_xy 
		sigma_xz = 2*self.mu*e_xz
		sigma_yz = 2*self.mu*e_yz

		return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz

	def predict(self, x, y, z):
		x = x.reshape(-1, 1)
		y = y.reshape(-1, 1)
		z = z.reshape(-1, 1)
		tx = create_tensor(x)
		ty = create_tensor(y)
		tz = create_tensor(z)
		deform = self.net([tx, ty, tz]).detach().numpy()

		return deform

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

