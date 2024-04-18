import sys
import numpy as np
sys.path.append('../pinns/hyper_elasticity_2D')
import neo_hookian as nh
import matplotlib.pyplot as plt
import torch
import seaborn as sns
fenics_info = np.loadtxt("../data/hyper_beam/hyper_beam_fenics_info.txt", delimiter=',')
fenics_beauty = np.loadtxt("../data/hyper_beam/hyper_beam_fenics_beauty.txt", delimiter=',')

exact = {'x': fenics_info[:, 0], 'y': fenics_info[:, 1], 'u_x': fenics_info[:, 2], 'u_y': fenics_info[:, 3], 'P_xx': fenics_info[:, 4], 'P_xy': fenics_info[:, 5], 'P_yx': fenics_info[:, 6], 'P_yy': fenics_info[:, 7]}

def plot_elasticity_pinn_and_exact():
	"""
	Plots the elasticity parameters for the pinn and the exact solution.
	"""
	nx = 100
	ny = 100
	x = np.linspace(0, 1, nx)
	y = np.linspace(0, 1, ny)
	xij, yij = np.meshgrid(x, y)
	x, y = xij.flatten(), yij.flatten()

	epochs = 200000

	# Creating models. 
	model_mixed = nh.Neo_hookian(10, 10, 4, 40, epochs)
	model_displacement = nh.Neo_hookian(10, 10, 4, 40, epochs, n_outputs = 2)

	# Solving models.
	model_mixed.solve()
	model_displacement.solve()

	# Predicting.
	mixed_pred = model_mixed.predict(xij, yij)
	displacement_pred = model_displacement.predict(xij, yij)
	
	mses_mixed_vs_no_mixed(model_mixed.losses, model_displacement.losses)

	# Printing the errors. 
	print(f'The MSE for u_x using Mixed PINNS is: {mean_squared_error(fenics_beauty[:, 2], mixed_pred[:, 0])}')
	print(f'The MSE for u_y using Mixed PINNS is: {mean_squared_error(fenics_beauty[:, 3], mixed_pred[:, 1])}')
	print(f'The MSE for s_xx using Mixed PINNS is: {mean_squared_error(fenics_beauty[:, 4], mixed_pred[:, 2])}')
	print(f'The MSE for s_xy using Mixed PINNS is: {mean_squared_error(fenics_beauty[:, 5], mixed_pred[:, 3])}')
	print(f'The MSE for s_yx using Mixed PINNS is: {mean_squared_error(fenics_beauty[:, 6], mixed_pred[:, 4])}')
	print(f'The MSE for s_yy using Mixed PINNS is: {mean_squared_error(fenics_beauty[:, 7], mixed_pred[:, 5])}')

	# Displacement. 
	print(f'The MSE for u_x using Displacement PINNS is: {mean_squared_error(fenics_beauty[:, 2], displacement_pred[:, 0])}')
	print(f'The MSE for u_y using Displacement PINNS is: {mean_squared_error(fenics_beauty[:, 3], displacement_pred[:, 1])}')
	print(f'The MSE for s_xx using Displacement PINNS is: {mean_squared_error(fenics_beauty[:, 4], displacement_pred[:, 2])}')
	print(f'The MSE for s_xy using Displacement PINNS is: {mean_squared_error(fenics_beauty[:, 5], displacement_pred[:, 3])}')
	print(f'The MSE for s_yx using Displacement PINNS is: {mean_squared_error(fenics_beauty[:, 6], displacement_pred[:, 4])}')
	print(f'The MSE for s_yy using Displacement PINNS is: {mean_squared_error(fenics_beauty[:, 7], displacement_pred[:, 5])}')

	create_comparison_plots(mixed_pred, displacement_pred, x, y, "plots_hyper_2D/comparison.png")
	

def mses_mixed_vs_no_mixed(mixed_losses, no_mixed_losses):
	epochs = np.arange(len(mixed_losses))
	plt.loglog(epochs, mixed_losses, label="Mixed PINN loss",alpha=0.5)
	plt.loglog(epochs, no_mixed_losses, label="No Mixed PINN loss",alpha=0.5)
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("MSE")
	plt.grid()
	plt.savefig("plots_hyper_2D/mixed_vs_nomixed_losses.pdf")
	plt.close()

def plot_losses(internal_losses, dirichlet_losses, neumann_losses, sigma_losses):
	epochs = np.arange(len(internal_losses))
	plt.loglog(epochs, internal_losses, label="Physics loss",alpha=0.5)
	plt.loglog(epochs, dirichlet_losses, label="Dirichlet loss",alpha=0.5)
	plt.loglog(epochs, neumann_losses, label="Neumann loss",alpha=0.5)
	plt.loglog(epochs, sigma_losses, label="Sigma losses",alpha=0.5)
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("MSE")
	plt.grid()
	plt.savefig("plots_linear_2D/losses.pdf")

def create_comparison_plots(pinns, nmixed_pinns, x, y, savepath):
	"""
	Create a 3x5 subplot layout to compare PINN, exact, and FEniCS results.

	Parameters:
	- pinns: Data from PINN simulations.
	- exact: Exact solution data.
	- fenics: Data from FEniCS simulations.
	- savepath: Where to save the figure.
	"""
	# Extract data

	pinn_u_x, pinn_u_y, pinn_s_xx, pinn_s_xy, pinn_s_yx, pinn_s_yy = pinns[:, 0], pinns[:, 1], pinns[:, 2], pinns[:, 3], pinns[:, 4], pinns[:, 5]
	fenics_u_x, fenics_u_y, fenics_s_xx, fenics_s_xy, fenics_s_yx, fenics_s_yy = fenics_beauty[:, 2], fenics_beauty[:, 3], fenics_beauty[:, 4], fenics_beauty[:, 5], fenics_beauty[:, 6], fenics_beauty[:, 7]

	# Maximums.
	xmin = 0
	xmax = 1
	ymin = 0
	ymax = 1

	data_list = [
		(nmixed_pinns[:, 0], 'PINN $u_x$'),
		(nmixed_pinns[:, 1], 'PINN $u_y$'),
		(nmixed_pinns[:, 2], 'PINN $s_{xx}$'),
		(nmixed_pinns[:, 3], 'PINN $s_{xy}$'),  # Assuming second variable is not used
		(nmixed_pinns[:, 4], 'PINN $s_{yx}$'),
		(nmixed_pinns[:, 5], 'PINN $s_{yy}$'),
		(pinn_u_x, 'Mixed PINN $u_x$'),
		(pinn_u_y, 'Mixed PINN $u_y$'),
		(pinn_s_xx, 'Mixed PINN $s_{xx}$'),
		(pinn_s_xy, 'Mixed PINN $s_{xy}$'),  # Assuming second variable is not used
		(pinn_s_yx, 'Mixed PINN $s_{yx}$'),  # Assuming second variable is not used
		(pinn_s_yy, 'Mixed PINN $s_{yy}$'),
		(fenics_u_x, 'FEM $u_x$'),
		(fenics_u_y, 'FEM $u_y$'),
		(fenics_s_xx, 'FEM $s_{xx}$'),
		(fenics_s_xy, 'FEM $s_{xy}$'),
		(fenics_s_yx, 'FEM $s_{yx}$'),
		(fenics_s_yy, 'FEM $s_{yy}$')
	]

	fig, axes = plt.subplots(3, 6, figsize=(13, 7))
	for ax, (u, title) in zip(axes.flatten(), data_list):
		cp = ax.scatter(x, y, c=u, alpha=0.7, cmap='jet', marker='o', s=int(2)) 
		fig.colorbar(cp, ax=ax)
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

def plot_inverse_elasticity():
	epochs = 100000
	epochs_list = np.arange(epochs)
	mu = 1.0
	kappa = 0.5
	seeds = [10, 123, 500]

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN by using stress data")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=kappa, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\kappa*$')
	colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))

	for s in range(len(seeds)):
		model = nh.Neo_hookian(10, 10, 4, 40, epochs, seed=seeds[s], problem='inverse', exact=exact)	
		model.solve()

		ax.plot(epochs_list, model.mus, color=colors[s], alpha=0.5)
		ax.plot(epochs_list, model.kappas, color=colors[s], alpha=0.5)

	print(f"The last mu was: {model.mus[-1]:.4f}")
	print(f"The last kappa was: {model.kappas[-1]:.4f}")
	
	ax.legend()
	plt.grid()
	plt.savefig('plots_hyper_2D/' + "inverse_parameters_stress.png", dpi=300)

def plot_inverse_fixed():
	mu = 1
	kappa = 0.5
	epochs = 200000
	epochs_list = np.arange(epochs)

	mu_fixed = nh.Neo_hookian(10, 10, 4, 40, epochs, fixed='mu', problem='inverse', exact=exact)	
	mu_fixed.solve()
	kappa_fixed = nh.Neo_hookian(10, 10, 4, 40, epochs, fixed='kappa', problem='inverse', exact=exact)	
	kappa_fixed.solve()

	fig, ax = plt.subplots(2, 1, figsize=(10, 10))

	ax[0].plot(epochs_list, mu_fixed.kappas, alpha=0.5)
	ax[0].hlines(y=kappa, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='Exact $\kappa$')
	ax[0].set_xlabel("Epochs")
	ax[0].set_title("$\mu$ fixed with stress data")
	ax[0].grid()
	ax[0].legend()

	ax[1].plot(epochs_list, kappa_fixed.mus, alpha=0.5)
	ax[1].hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='Exact $\mu$')
	ax[1].set_xlabel("Epochs")
	ax[1].set_title("$\kappa$ fixed with stress data")
	ax[1].grid()
	ax[1].legend()

	print(f"The last kappa was: {mu_fixed.kappas[-1]:.4f}")
	print(f"The last mu was: {kappa_fixed.mus[-1]:.4f}")

	plt.savefig("plots_hyper_2D/fixed_inverse.pdf")

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
	plot_elasticity_pinn_and_exact()
	#plot_inverse_elasticity()
	#plot_inverse_fixed()







