import sys
import numpy as np
sys.path.append('../pinns/linear_elasticity_2D')
import clamped_elasticity_pytorch as cep
import clamped_elasticity_inverse_pytorch as ceip
import matplotlib.pyplot as plt
import torch
import seaborn as sns
fenics = np.loadtxt("../data/clamped_beam/clamped_beam_fenics_beauty.txt", delimiter=',')

def plot_elasticity_pinn_and_exact():
	"""
	Plots the elasticity parameters for the pinn and the exact solution.
	"""
	nx = 200
	ny = 200
	x = np.linspace(0, 1, nx)
	y = np.linspace(0, 1, ny)
	xij, yij = np.meshgrid(x, y)
	x, y = xij.flatten(), yij.flatten()
	k = 1
	mu = 1
	lambda_ = 0.5

	# Exact.
	exact = u_exact(x, y, k, mu, lambda_)

	# Predicting. 
	network, internal_losses, dirichlet_losses, neumann_losses, sigma_losses = cep.manu_elasticity_solve(20, 20, 4, 40, 200000, 1e-3, k=k, verbose=True)
	pinns = cep.predict(network, xij, yij)

	# Printing the errors. 
	print(f'The MSE for u_x using PINNS is: {mean_squared_error(exact["u_x"], pinns[:, 0])}')
	print(f'The MSE for u_y using PINNS is: {mean_squared_error(exact["u_y"], pinns[:, 1])}')
	print(f'The MSE for s_xx using PINNS is: {mean_squared_error(exact["s_xx"], pinns[:, 2])}')
	print(f'The MSE for s_yy using PINNS is: {mean_squared_error(exact["s_yy"], pinns[:, 3])}')
	print(f'The MSE for s_xy using PINNS is: {mean_squared_error(exact["s_xy"], pinns[:, 4])}')

	# Printing the errors. 
	print(f'The MSE for u_x using FEM is: {mean_squared_error(exact["u_x"], fenics[:, 0])}')
	print(f'The MSE for u_y using FEM is: {mean_squared_error(exact["u_y"], fenics[:, 1])}')
	print(f'The MSE for s_xx using FEM is: {mean_squared_error(exact["s_xx"], fenics[:, 2])}')
	print(f'The MSE for s_yy using FEM is: {mean_squared_error(exact["s_yy"], fenics[:, 3])}')
	print(f'The MSE for s_xy using FEM is: {mean_squared_error(exact["s_xy"], fenics[:, 4])}')

	create_comparison_plots(pinns, exact, fenics, x, y, "plots_linear_2D/comparison.png")
	plot_losses(internal_losses, dirichlet_losses, neumann_losses, sigma_losses)
	

def plot_losses(internal_losses, dirichlet_losses, neumann_losses, sigma_losses):
	epochs = np.arange(len(internal_losses))
	plt.loglog(epochs, internal_losses, label="Physics loss")
	plt.loglog(epochs, dirichlet_losses, label="Dirichlet loss")
	plt.loglog(epochs, neumann_losses, label="Neumann loss")
	plt.loglog(epochs, sigma_losses, label="Sigma losses")
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("MSE")
	plt.savefig("plots_linear_2D/losses.pdf")

def create_comparison_plots(pinns, exact, fenics, x, y, savepath):
	"""
	Create a 3x5 subplot layout to compare PINN, exact, and FEniCS results.

	Parameters:
	- pinns: Data from PINN simulations.
	- exact: Exact solution data.
	- fenics: Data from FEniCS simulations.
	- savepath: Where to save the figure.
	"""
	# Extract data
	pinn_u_x, pinn_u_y, pinn_s_xx, pinn_s_yy, pinn_s_xy = pinns[:, 0], pinns[:, 1], pinns[:, 2], pinns[:, 3], pinns[:, 4]
	exact_u_x, exact_u_y, exact_s_xx, exact_s_yy, exact_s_xy = exact['u_x'], exact['u_y'], exact['s_xx'], exact['s_yy'], exact['s_xy']
	fenics_u_x, fenics_u_y, fenics_s_xx, fenics_s_yy, fenics_s_xy = fenics[:, 0], fenics[:, 1], fenics[:, 2], fenics[:, 3], fenics[:, 4]

	# Maximums.
	xmin = 0
	xmax = 1
	ymin = 0
	ymax = 1

	data_list = [
		(pinn_u_x, 'PINN $u_x$'),
		(pinn_u_y, 'PINN $u_y$'),
		(pinn_s_xx, 'PINN $s_{xx}$'),
		(pinn_s_xy, 'PINN $s_{xy}$'),  # Assuming second variable is not used
		(pinn_s_yy, 'PINN $s_{yy}$'),
		(exact_u_x, 'Exact $u_x$'),
		(exact_u_y, 'Exact $u_y$'),
		(exact_s_xx, 'Exact $s_{xx}$'),
		(exact_s_xy, 'Exact $s_{xy}$'),
		(exact_s_yy, 'Exact $s_{yy}$'),
		(fenics_u_x, 'FEniCS $u_x$'),
		(fenics_u_y, 'FEniCS $u_y$'),
		(fenics_s_xx, 'FEniCS $s_{xx}$'),
		(fenics_s_xy, 'FEniCS $s_{xy}$'),
		(fenics_s_yy, 'FEniCS $s_{yy}$')
	]

	fig, axes = plt.subplots(3, 5, figsize=(7, 5))
	for ax, (u, title) in zip(axes.flatten(), data_list):
		cp = ax.scatter(x, y, c=u, alpha=0.7, cmap='jet', marker='o', s=int(2)) 
		ax.set_title(title)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlim([xmin, xmax])
		ax.set_ylim([ymin, ymax])
		ax.set_xlabel("x (m)")
		ax.set_ylabel("y (m)")

	plt.tight_layout()
	plt.savefig(savepath, dpi=300)
	plt.close()

def plot_inverse_elasticity_stress():
	k = 1
	mu = 1
	lambda_ = 0.5
	seeds = [10, 123, 500, 1000, 2024]
	epochs = 15000
	epochs_list = np.arange(epochs)

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN by using stress data")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=lambda_, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\lambda*$')
	colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))

	for s in range(len(seeds)):
		mus, lambdas = ceip.manu_elasticity_inverse_solve(20, 20, 4, 40, epochs, 1e-3, k=k, verbose=True, seed=seeds[s])	
		ax.plot(epochs_list, mus, color=colors[s],alpha=0.5)
		ax.plot(epochs_list, lambdas, color=colors[s],alpha=0.5)
	
	ax.legend()
	plt.savefig('plots_linear_2D/' + "inverse_parameters_stress.png", dpi=300)

def plot_inverse_elasticity_strain():
	k = 1
	mu = 1
	lambda_ = 0.5
	seeds = [10, 123, 500, 1000, 2024]
	epochs = 15000
	epochs_list = np.arange(epochs)

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN by using strain data")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=lambda_, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\lambda*$')
	colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))

	for s in range(len(seeds)):
		mus, lambdas = ceip.manu_elasticity_inverse_solve(20, 20, 4, 40, epochs, 1e-3, k=k, verbose=True, exact_data_type='strain', seed=seeds[s])	
		ax.plot(epochs_list, mus, color=colors[s],alpha=0.5)
		ax.plot(epochs_list, lambdas, color=colors[s],alpha=0.5)
	
	ax.legend()
	plt.savefig('plots_linear_2D/' + "inverse_parameters_strain.pdf", dpi=300)

def plot_frequency_change():
	ks = [1, 3, 5, 7, 9, 11]
	epochs = 15000

	nx = 200
	ny = 200
	x = np.linspace(0, 1, nx)
	y = np.linspace(0, 1, ny)
	xij, yij = np.meshgrid(x, y)
	x, y = xij.flatten(), yij.flatten()
	mu = 1
	lambda_ = 0.5
	colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))

	# Predicting. 
	mses_ux = []
	mses_uy = []
	mses_sxx = []
	mses_syy = []
	mses_sxy = []

	for i in range(len(ks)):
		exact = u_exact(x, y, ks[i], mu, lambda_)
		network = cep.manu_elasticity_solve(20, 20, 4, 40, epochs, 1e-3, k=ks[i])
		pinns = cep.predict(network, xij, yij)
		mses_ux.append(mean_squared_error(exact["u_x"], pinns[:, 0]))
		mses_uy.append(mean_squared_error(exact["u_y"], pinns[:, 1]))

	plt.loglog(ks, mses_ux, 'ro-', color=colors[0], label='$u_x$')
	plt.loglog(ks, mses_uy, 'ro-', color=colors[1], label='$u_y$')
	plt.ylabel('$MSE_{log}$')
	plt.xlabel('Value of k')
	plt.legend()
	plt.savefig('plots_linear_2D/' + "frquency_change.pdf", dpi=300)

def contour_plot(x, y, u, title, savepath):
	# Maximums.
	xmin = 0
	xmax = 1
	ymin = 0
	ymax = 1

	# plot PINN results
	fig11, ax11 = plt.subplots()
	ax11.set_aspect('equal')
	cp = ax11.scatter(x, y, c=u, alpha=0.7, cmap='jet', marker='o', s=int(2))   
	ax11.set_xticks([])
	ax11.set_yticks([])
	ax11.set_xlim([xmin, xmax])
	ax11.set_ylim([ymin, ymax])
	ax11.set_xlabel("x (m)")
	ax11.set_ylabel("y (m)")
	plt.title(title)
	fig11.colorbar(cp)
	plt.savefig('plots_linear_2D/' + savepath, dpi=300)

def u_exact(x, y, k, mu, lambd):
	# Constans. 
	pi = np.pi
	cos = np.cos
	sin = np.sin

	u_x = (np.sin(k*np.pi*x))
	u_y = (np.sin(k*np.pi*x)*np.cos(k*np.pi*y))

	e_xx = (pi*k*cos(pi*k*x))
	e_yy = (-pi*k*sin(pi*k*x)*sin(pi*k*y))
	e_xy = (0.5*pi*k*cos(pi*k*x)*cos(pi*k*y))

	s_xx = (2*pi*k*mu*cos(pi*k*x) + lambd*(-pi*k*sin(pi*k*x)*sin(pi*k*y) + pi*k*cos(pi*k*x)))
	s_yy = (-2*pi*k*mu*sin(pi*k*x)*sin(pi*k*y) + lambd*(-pi*k*sin(pi*k*x)*sin(pi*k*y) + pi*k*cos(pi*k*x)))
	s_xy = (1.0*pi*k*mu*cos(pi*k*x)*cos(pi*k*y))

	exact = {"u_x": u_x, "u_y": u_y, "e_xx": e_xx, "e_yy": e_yy, "e_xy": e_xy, "s_xx": s_xx, "s_yy": s_yy, "s_xy": s_xy}

	return exact

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """
    Calculate the R-squared score.
    
    Parameters:
    - y_true: numpy array, true values.
    - y_pred: numpy array, predicted values.
    
    Returns:
    - r2: float, the R-squared score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


if __name__ == '__main__':
	#plot_elasticity_pinn_and_exact()
	#plot_inverse_elasticity_stress()
	#plot_inverse_elasticity_strain()
	plot_frequency_change()
	#plot_inverse_with_boundaries()







