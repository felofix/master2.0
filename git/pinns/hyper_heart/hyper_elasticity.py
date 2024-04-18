import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim import lr_scheduler
import torch.nn.init as init
import matplotlib.pyplot as plt

Pi = np.pi

class Neo_Hookian:
	def __init__(self, model, n_hid, n_neu, epochs, problem = 'forward', \
				 lr=1e-3, activation_function = nn.Tanh(),\
				 n_inputs = 3, n_outputs = 12, exact = None, seed=1234, asf = None, Ta = 1):

		# asf = active strain formulation

		torch.manual_seed(seed)
		np.random.seed(seed)

		# Model specifics. 
		self.problem = problem
		self.exact = exact
		self.afs = afs
		self.Ta = Ta

		# Neural network.
		self.net = Net(n_hid, n_neu, n_inputs, n_outputs, activation_function)
		self.net = self.net.to(device)
		self.epochs = epochs

		if self.problem == "forward":
			self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
			self.mu = 1
			self.kappa = 0.5

		elif self.problem == "inverse":
			self.mu = nn.Parameter(torch.tensor(float(np.random.randint(5)), requires_grad = True))
			self.kappa = nn.Parameter(torch.tensor(float(np.random.randint(5)), requires_grad = True))
			self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.mu] + [self.kappa], lr=lr)

		# lists of things.
		self.losses = []
		self.mus = []
		self.kappas = []

		self.exact = exact

		# Defining boundaries and such. 
		self.c = model.coordinates # All coordinates. 
		self.bc = model.bc  # Boundary coordinates.
		self.bi = model.bi  # Boundary indeces. 
		self.bn = model.bn  # Boundary normals. 

		self.X, self.Y, self.Z = self.c[:, 0], self.c[:, 1], self.c[:, 2]

	def solve(self):
		"""
		Solving specific problem. 
		"""
		if self.problem == 'inverse':
			self.X, self.Y, self.Z = self.exact['x'], self.exact['y'], self.exact['z']

		for epoch in range(self.epochs):
			if self.problem == 'forward':
				loss = self.forward_loss()
			elif self.problem == 'inverse':
				loss = self.inverse_loss()
				self.mus.append(self.mu.item())
				self.kappas.append(self.kappa.item())

			print(loss.item(), epoch)
			self.losses.append(loss.item())
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

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

		FT = [[F[0][0], F[1][0], F[2][0]],
			  [F[0][1], F[1][1], F[2][1]],
    		  [F[0][2], F[1][2], F[2][2]]]

		Finv  = inverse_matrix(F)

		FTinv = [[Finv[0][0], Finv[1][0], Finv[2][0]],
		     	 [Finv[0][1], Finv[1][1], Finv[2][1]],
		     	 [Finv[0][2], Finv[1][2], Finv[2][2]]]

		Cxx = FT[0][0] * F[0][0] + FT[0][1] * F[1][0] + FT[0][2] * F[2][0]
		Cxy1= FT[0][0] * F[0][1] + FT[0][1] * F[1][1] + FT[0][2] * F[2][1]
		Cxz = FT[0][0] * F[0][2] + FT[0][1] * F[1][2] + FT[0][2] * F[2][2]

		Cyx = FT[1][0] * F[0][0] + FT[1][1] * F[1][0] + FT[1][2] * F[2][0]
		Cyy = FT[1][0] * F[0][1] + FT[1][1] * F[1][1] + FT[1][2] * F[2][1]
		Cyz = FT[1][0] * F[0][2] + FT[1][1] * F[1][2] + FT[1][2] * F[2][2]

		Czx= FT[2][0] * F[0][0] + FT[2][1] * F[1][0] + FT[2][2] * F[2][0]
		Czy = FT[2][0] * F[0][1] + FT[2][1] * F[1][1] + FT[2][2] * F[2][1]
		Czz = FT[2][0] * F[0][2] + FT[2][1] * F[1][2] + FT[2][2] * F[2][2]

		C = [[Cxx, Cxy, Cxz],
			 [Cyx, Cyy, Cyz],
			 [Czx, Czy, Czz]]

		active_strain_f = Ta*self.active_strain(C)

		PK_xx = self.mu*F[0][0] - self.mu*FTinv[0][0] + self.kappa*torch.log(J)*FTinv[0][0] + active_strain_f
		PK_yy = self.mu*F[1][1] - self.mu*FTinv[1][1] + self.kappa*torch.log(J)*FTinv[1][1] + active_strain_f
		PK_zz = self.mu*F[2][2] - self.mu*FTinv[2][2] + self.kappa*torch.log(J)*FTinv[2][2] + active_strain_f

		PK_xy = self.mu*F[0][1] - self.mu*FTinv[0][1] + self.kappa*torch.log(J)*FTinv[0][1] + active_strain_f
		PK_yx = self.mu*F[1][0] - self.mu*FTinv[1][0] + self.kappa*torch.log(J)*FTinv[1][0] + active_strain_f

		PK_xz = self.mu*F[0][2] - self.mu*FTinv[0][2] + self.kappa*torch.log(J)*FTinv[0][2] + active_strain_f
		PK_zx = self.mu*F[2][0] - self.mu*FTinv[2][0] + self.kappa*torch.log(J)*FTinv[2][0] + active_strain_f

		PK_yz = self.mu*F[1][2] - self.mu*FTinv[1][2] + self.kappa*torch.log(J)*FTinv[1][2] + active_strain_f
		PK_zy = self.mu*F[2][1] - self.mu*FTinv[2][1] + self.kappa*torch.log(J)*FTinv[2][1] + active_strain_f

		return PK_xx, PK_yy, PK_zz, PK_xy, PK_yx, PK_xz, PK_zx, PK_yz, PK_zy

	def active_strain(self, C):
		if self.asf == None:
			return 0 
		# Create C.
		else:
			fx = create_tensor(self.asf[:, 0].reshape(-1, 1))
			fy = create_tensor(self.asf[:, 1].reshape(-1, 1))
			fz = create_tensor(self.asf[:, 2].reshape(-1, 1))

			astrain = fx**2*C[0][0] + 2*fx*fy*C[0][1] + 2*fx*fz*C[0][2] + fy**2*C[1][1] + 2*fy*fz*C[1][2] + fz**2*C[2][2]

			return astrain



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




