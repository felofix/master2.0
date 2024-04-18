import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
import torch.nn.init as init
import matplotlib.pyplot as plt
from scipy.stats.qmc import Halton

torch.manual_seed(1234)
np.random.seed(1234)

Pi = np.pi

class Neo_Hookian_3D:
	def __init__(self, n_hid, n_neu, epochs, problem = 'forward', \
				 lr=1e-3, activation_function = nn.Tanh(),\
				 n_inputs = 3, n_outputs = 12, exact = None):

		# Model specifics. 
		self.problem = problem
		self.exact = exact

		# Neural network.
		self.net = Net(n_hid, n_neu, n_inputs, n_outputs, activation_function)
		self.net = self.net.to(device)
		self.epochs = epochs

		if self.problem == "forward":
			self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
			self.mu = 1.0
			self.kappa = 0.5

		elif self.problem == "inverse":
			self.mu = nn.Parameter(torch.tensor(3.0, requires_grad = True))
			self.kappa = nn.Parameter(torch.tensor(5.0, requires_grad = True))
			self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.mu] + [self.kappa], lr=lr)
	
		# lists of things.
		self.losses = []
		self.mus = []
		self.kappas = []

		# Dimension
		self.nx = 10
		self.ny = 10
		self.nz = 10

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

		all_side_indices = np.concatenate([self.left[0], self.right[0], \
		    							   self.bottom[0], self.top[0], \
		    							   self.front[0], self.back[0]])

		all_indices = np.arange(len(self.X))
		self.internal = np.setdiff1d(all_indices, all_side_indices)

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
		
		else:
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
				self.mus.append(self.mu.item())
				self.kappas.append(self.kappa.item())

			if epoch % 1000 == 0:
				print(epoch)
				print(loss.item())
				
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

		# Predicted values. 
		u = self.net([X, Y, Z])
		u_X =  u[:, 0].reshape(-1, 1)
		u_Y =  u[:, 1].reshape(-1, 1)
		u_Z =  u[:, 2].reshape(-1, 1)
		p_xx = u[:, 3].reshape(-1, 1)
		p_yy = u[:, 4].reshape(-1, 1)
		p_zz = u[:, 5].reshape(-1, 1)
		p_xy = u[:, 6].reshape(-1, 1)
		p_yx = u[:, 7].reshape(-1, 1)
		p_xz = u[:, 8].reshape(-1, 1)
		p_zx = u[:, 9].reshape(-1, 1)
		p_yz = u[:, 10].reshape(-1, 1)
		p_zy = u[:, 11].reshape(-1, 1)

		# Calculated Piola - Kirchhoff stresses.
		# Insert piola calculation.
		PK_xx, PK_yy, PK_zz, PK_xy, PK_yx, PK_xz, PK_zx, PK_yz, PK_zy = self.piola_kirchhoff(u_X, u_Y, u_Z, X, Y, Z)

		# Sigma losses.
		pxx_loss = torch.mean(torch.square(PK_xx - p_xx))
		pyy_loss = torch.mean(torch.square(PK_yy - p_yy))
		pzz_loss = torch.mean(torch.square(PK_zz - p_zz))
		pxy_loss = torch.mean(torch.square(PK_xy - p_xy))
		pyx_loss = torch.mean(torch.square(PK_yx - p_yx))
		pxz_loss = torch.mean(torch.square(PK_xz - p_xz))
		pzx_loss = torch.mean(torch.square(PK_zx - p_zx))
		pyz_loss = torch.mean(torch.square(PK_yz - p_yz))
		pzy_loss = torch.mean(torch.square(PK_zy - p_zy))

		piola_loss = pxx_loss + pyy_loss + pzz_loss + pxy_loss + pyx_loss + pxz_loss + pzx_loss + pyz_loss + pzy_loss

		# Internal losses. 
		piola_x = diff(p_xx, X) + diff(p_xy, Y) + diff(p_xz, Z)
		piola_y = diff(p_yx, X) + diff(p_yy, Y) + diff(p_yz, Z)
		piola_z = diff(p_zx, X) + diff(p_zy, Y) + diff(p_zz, Z)
		piola_x_loss = torch.mean(torch.square(piola_x))
		piola_y_loss = torch.mean(torch.square(piola_y)) 
		piola_z_loss = torch.mean(torch.square(piola_z)) 
		internal_loss = piola_x_loss + piola_y_loss + piola_z_loss

		# Boundary losses, should be changes it can taake general boundary conditions. 
		# Dirichlet loss. 
		D_X_prep = torch.cat([u_X[self.left]], dim=0)
		D_Y_prep = torch.cat([u_Y[self.left]], dim=0)
		D_Z_prep = torch.cat([u_Z[self.left]], dim=0)
		dirichlet_loss = torch.mean(torch.square(D_X_prep)) + torch.mean(torch.square(D_Y_prep)) + torch.mean(torch.square(D_Z_prep))
		
		# Neumann loss. 
		tx_front, ty_front, tz_front = self.create_traction(self.front, self.front_normal, 0.0, p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy)
		tx_back, ty_back, tz_back = self.create_traction(self.back, self.back_normal, 0.0, p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy)
		tx_top, ty_top, tz_top = self.create_traction(self.top, self.top_normal, -0.3, p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy)
		tx_bottom, ty_bottom, tz_bottom = self.create_traction(self.bottom, self.bottom_normal, 0.0, p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy)
		tx_right, ty_right, tz_right = self.create_traction(self.right, self.right_normal, 0.0, p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy)

		t_x_prep = torch.cat([tx_front, tx_back, tx_bottom, tx_top, tx_right], dim=0)
		t_y_prep = torch.cat([ty_front, ty_back, ty_bottom, ty_top, ty_right], dim=0)
		t_z_prep = torch.cat([tz_front, tz_back, tz_bottom, tz_top, tz_right], dim=0)

		neumann_loss = torch.mean(torch.square(t_x_prep)) + torch.mean(torch.square(t_y_prep)) + torch.mean(torch.square(t_z_prep))

		total_loss = internal_loss + piola_loss + neumann_loss*10 + dirichlet_loss*2

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
		p_xx = u[:, 3].reshape(-1, 1)
		p_yy = u[:, 4].reshape(-1, 1)
		p_zz = u[:, 5].reshape(-1, 1)
		p_xy = u[:, 6].reshape(-1, 1)
		p_yx = u[:, 7].reshape(-1, 1)
		p_xz = u[:, 8].reshape(-1, 1)
		p_zx = u[:, 9].reshape(-1, 1)
		p_yz = u[:, 10].reshape(-1, 1)
		p_zy = u[:, 11].reshape(-1, 1)

		# Calculated Piola - Kirchhoff stresses.
		# Insert piola calculation.
		PK_xx, PK_yy, PK_zz, PK_xy, PK_yx, PK_xz, PK_zx, PK_yz, PK_zy = self.piola_kirchhoff(u_X, u_Y, u_Z, X, Y, Z)

		# Sigma losses.
		pxx_loss = torch.mean(torch.square(PK_xx - p_xx))
		pyy_loss = torch.mean(torch.square(PK_yy - p_yy))
		pzz_loss = torch.mean(torch.square(PK_zz - p_zz))
		pxy_loss = torch.mean(torch.square(PK_xy - p_xy))
		pyx_loss = torch.mean(torch.square(PK_yx - p_yx))
		pxz_loss = torch.mean(torch.square(PK_xz - p_xz))
		pzx_loss = torch.mean(torch.square(PK_zx - p_zx))
		pyz_loss = torch.mean(torch.square(PK_yz - p_yz))
		pzy_loss = torch.mean(torch.square(PK_zy - p_zy))

		piola_loss = pxx_loss + pyy_loss + pzz_loss + pxy_loss + pyx_loss + pxz_loss + pzx_loss + pyz_loss + pzy_loss

		# Internal losses. 
		piola_x = diff(p_xx, X) + diff(p_xy, Y) + diff(p_xz, Z)
		piola_y = diff(p_yx, X) + diff(p_yy, Y) + diff(p_yz, Z)
		piola_z = diff(p_zx, X) + diff(p_zy, Y) + diff(p_zz, Z)
		piola_x_loss = torch.mean(torch.square(piola_x))
		piola_y_loss = torch.mean(torch.square(piola_y)) 
		piola_z_loss = torch.mean(torch.square(piola_z)) 
		internal_loss = piola_x_loss + piola_y_loss + piola_z_loss
		
		# Boundary losses, should be changes it can taake general boundary conditions. 
		# Exact loss. 
		u_x_loss = torch.mean(torch.square(u_X - create_tensor(self.exact['u_x'])))
		u_y_loss = torch.mean(torch.square(u_Y - create_tensor(self.exact['u_y'])))
		u_z_loss = torch.mean(torch.square(u_Z - create_tensor(self.exact['u_z'])))

		exact_pxx_loss = torch.mean(torch.square(p_xx - create_tensor(self.exact['P_xx'])))
		exact_pyy_loss = torch.mean(torch.square(p_yy - create_tensor(self.exact['P_yy'])))
		exact_pzz_loss = torch.mean(torch.square(p_zz - create_tensor(self.exact['P_zz'])))
		exact_pxy_loss = torch.mean(torch.square(p_xy - create_tensor(self.exact['P_xy'])))
		exact_pyx_loss = torch.mean(torch.square(p_yx - create_tensor(self.exact['P_yx'])))
		exact_pxz_loss = torch.mean(torch.square(p_xz - create_tensor(self.exact['P_xz'])))
		exact_pzx_loss = torch.mean(torch.square(p_zx - create_tensor(self.exact['P_zx'])))
		exact_pyz_loss = torch.mean(torch.square(p_yz - create_tensor(self.exact['P_yz'])))
		exact_pzy_loss = torch.mean(torch.square(p_zy - create_tensor(self.exact['P_zy'])))

		exact_loss = u_x_loss + u_y_loss + u_z_loss + exact_pxx_loss + exact_pyy_loss + exact_pzz_loss + exact_pxy_loss + exact_pyx_loss + \
					 exact_pxz_loss + exact_pzx_loss + exact_pyz_loss + exact_pzy_loss

		total_loss = exact_loss + internal_loss + piola_loss

		return total_loss

	def create_traction(self, bc, normal, t, p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy):
		t_x = p_xx[bc]*normal[0] + p_xy[bc]*normal[1] + p_xz[bc]*normal[2]
		t_y = p_yx[bc]*normal[0] + p_yy[bc]*normal[1] + p_yz[bc]*normal[2]
		t_z = p_zx[bc]*normal[0] + p_zy[bc]*normal[1] + p_zz[bc]*normal[2]

		t_x_loss = t_x - t*normal[0]
		t_y_loss = t_y - t*normal[1]
		t_z_loss = t_z - t*normal[2]
		return t_x_loss, t_y_loss, t_z_loss
		

	def piola_kirchhoff(self, ux, uy, uz, X, Y, Z):
		"""
		Calculates the deformation gradient using autograd. Changed grad too diff 
		when I should actually create the function. 
		"""
		ux_x = diff(ux, X) + 1
		ux_y = diff(ux, Y)
		ux_z = diff(ux, Z)

		uy_x = diff(uy, X)
		uy_y = diff(uy, Y) + 1
		uy_z = diff(uy, Z)

		uz_x = diff(uz, X)
		uz_y = diff(uz, Y)
		uz_z = diff(uz, Z) + 1

		F = [[ux_x, ux_y, ux_z],
		     [uy_x, uy_y, uy_z],
		     [uz_x, uz_y, uz_z]]

		J = (F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1]) - \
             F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0]) + \
             F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]))

		Finv  = inverse_matrix(F)

		FTinv = [[Finv[0][0], Finv[1][0], Finv[2][0]],
		     	 [Finv[0][1], Finv[1][1], Finv[2][1]],
		     	 [Finv[0][2], Finv[1][2], Finv[2][2]]]

		PK_xx = self.mu*F[0][0] - self.mu*FTinv[0][0] + self.kappa*torch.log(J)*FTinv[0][0]
		PK_yy = self.mu*F[1][1] - self.mu*FTinv[1][1] + self.kappa*torch.log(J)*FTinv[1][1]
		PK_zz = self.mu*F[2][2] - self.mu*FTinv[2][2] + self.kappa*torch.log(J)*FTinv[2][2]

		PK_xy = self.mu*F[0][1] - self.mu*FTinv[0][1] + self.kappa*torch.log(J)*FTinv[0][1]
		PK_yx = self.mu*F[1][0] - self.mu*FTinv[1][0] + self.kappa*torch.log(J)*FTinv[1][0]

		PK_xz = self.mu*F[0][2] - self.mu*FTinv[0][2] + self.kappa*torch.log(J)*FTinv[0][2]
		PK_zx = self.mu*F[2][0] - self.mu*FTinv[2][0] + self.kappa*torch.log(J)*FTinv[2][0]

		PK_yz = self.mu*F[1][2] - self.mu*FTinv[1][2] + self.kappa*torch.log(J)*FTinv[1][2]
		PK_zy = self.mu*F[2][1] - self.mu*FTinv[2][1] + self.kappa*torch.log(J)*FTinv[2][1]

		return PK_xx, PK_yy, PK_zz, PK_xy, PK_yx, PK_xz, PK_zx, PK_yz, PK_zy

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

def inverse_matrix(F):
    # Step 1: Calculate the determinant of F
    detF = (F[0][0] * (F[1][1]*F[2][2] - F[2][1]*F[1][2]) -
            F[0][1] * (F[1][0]*F[2][2] - F[1][2]*F[2][0]) +
            F[0][2] * (F[1][0]*F[2][1] - F[1][1]*F[2][0]))

    # Step 2 & 3: Calculate matrix of cofactors
    cofactors = []
    for row in range(3):
        cofactorRow = []
        for col in range(3):
            minor = [[F[i][j] for j in range(3) if j != col] for i in range(3) if i != row]
            # Calculate the determinant of the minor
            minorDet = minor[0][0]*minor[1][1] - minor[0][1]*minor[1][0]
            cofactor = (-1)**(row+col) * minorDet
            cofactorRow.append(cofactor)
        cofactors.append(cofactorRow)

    # Step 4: Calculate the adjugate of F (transpose of the matrix of cofactors)
    adjugate = [[cofactors[j][i] for j in range(3)] for i in range(3)]

    # Step 5: Calculate the inverse of F
    inverseF = [[adjugate[i][j] / detF for j in range(3)] for i in range(3)]
    return inverseF

def create_tensor(k):
	tensor = torch.tensor(k, dtype=torch.float32, requires_grad=True).to(device)
	return tensor

def diff(u, d):
	return torch.autograd.grad(u, d, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]