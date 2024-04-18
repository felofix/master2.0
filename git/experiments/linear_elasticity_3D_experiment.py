import sys
import numpy as np
sys.path.append('../pinns/linear_elasticity_3D')
import linear_3d as l3d
import matplotlib.pyplot as plt
import torch

data = np.loadtxt("../data/clamped_beam_3D/linear_data_fenics.txt", delimiter=',')
exact = {"x": data[:, 0].reshape(-1, 1), "y": data[:, 1].reshape(-1, 1), "z": data[:, 2].reshape(-1, 1),\
		 "u_x": data[:, 3].reshape(-1, 1), "u_y": data[:, 4].reshape(-1, 1), "u_z": data[:, 5].reshape(-1, 1),\
		 "s_xx": data[:, 6].reshape(-1, 1), "s_yy": data[:, 7].reshape(-1, 1), "s_zz": data[:, 8].reshape(-1, 1),\
		 "s_xy": data[:, 9].reshape(-1, 1), "s_xz": data[:, 10].reshape(-1, 1), "s_yz": data[:, 11].reshape(-1, 1), \
		 "u_xx": data[:, 12].reshape(-1, 1), "u_yy": data[:, 13].reshape(-1, 1), "u_zz": data[:, 14].reshape(-1, 1),\
		 "u_xy": data[:, 15].reshape(-1, 1), "u_yx": data[:, 16].reshape(-1, 1), "u_xz": data[:, 17].reshape(-1, 1), \
		 "u_zx": data[:, 18].reshape(-1, 1), "u_yz": data[:, 19].reshape(-1, 1), "u_zy": data[:, 20].reshape(-1, 1)}

def create_elasticity_pinn_data():
	epochsn = 400000
	epochs = np.arange(epochsn)
	model = l3d.Linear3D(4, 40, epochsn, problem='forward', exact=exact)
	model.solve()

	X, Y, Z = model.X, model.Y, model.Z

	u = model.predict(X, Y, Z)
	ux, uy, uz, sxx, syy, szz, sxy, sxz, syz = u[:, 0].reshape(-1, 1), u[:, 1].reshape(-1, 1), u[:, 2].reshape(-1, 1), u[:, 3].reshape(-1, 1), \
											   u[:, 4].reshape(-1, 1), u[:, 5].reshape(-1, 1), u[:, 6].reshape(-1, 1), u[:, 7].reshape(-1, 1), \
											   u[:, 8].reshape(-1, 1)

	with open('../data/clamped_beam_3D/linear_data_pinns.txt', 'w') as file:
		for i in range(len(sxx)):
			file.write(f'{X[i][0]}, {Y[i][0]}, {Z[i][0]}, {ux[i][0]}, {uy[i][0]}, {uz[i][0]}, {abs(uz[i] - exact["u_z"][i])[0]} \n')

	# Printing the errors. 
	print(f'The MSE for u_x using Mixed PINNS is: {mean_squared_error(exact["u_x"], ux)}')
	print(f'The MSE for u_y using Mixed PINNS is: {mean_squared_error(exact["u_y"], uy)}')
	print(f'The MSE for u_y using Mixed PINNS is: {mean_squared_error(exact["u_z"], uz)}')

	print(f'The MSE for s_xx using Mixed PINNS is: {mean_squared_error(exact["s_xx"], sxx)}')
	print(f'The MSE for s_yy using Mixed PINNS is: {mean_squared_error(exact["s_yy"], syy)}')
	print(f'The MSE for s_xy using Mixed PINNS is: {mean_squared_error(exact["s_zz"], szz)}')

	print(f'The MSE for s_xx using Mixed PINNS is: {mean_squared_error(exact["s_xy"], sxy)}')
	print(f'The MSE for s_yy using Mixed PINNS is: {mean_squared_error(exact["s_xz"], sxz)}')
	print(f'The MSE for s_xy using Mixed PINNS is: {mean_squared_error(exact["s_yz"], syz)}')

	plot_loss(model.losses)

def plot_loss(mses):
	epochs = np.arange(len(mses))
	plt.loglog(epochs, mses, label="Total loss", alpha=0.5)
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("MSE")
	plt.grid()
	plt.savefig("plots_linear_3D/losses.pdf")

def plot_inverse_elasticity():
	mu = 1
	lambda_ = 0.5

	epochsn = 200000
	epochs = np.arange(epochsn)
	model = l3d.Linear3D(4, 40, epochsn, problem='inverse', exact=exact)
	model.solve()

	print(f'The last measured lambda value was: {model.lambdas[-1]:.2f}')
	print(f'The last measured mu value was: {model.mu[-1]:.2f}')

	fig, ax = plt.subplots()
	ax.plot(epochs, model.lambdas, alpha=0.5, label='$\lambda$')
	ax.plot(epochs, model.mus, alpha=0.5, label='$\mu$')
	ax.hlines(y=lambda_, xmin=0, xmax=len(epochs), linewidth=1, color='black', linestyle='dashed', label='Exact $\lambda$')
	ax.hlines(y= mu, xmin=0, xmax=len(epochs), linewidth=1, color='black', linestyle='dashed', label='Exact $\mu$')
	ax.set_xlabel("Epochs")
	ax.set_title("$\mu$ fixed with stress data")
	ax.grid()
	ax.legend()
	plt.savefig("plots_linear_3D/inverse.pdf")

def plot_inverse_fixed():
	mu = 1
	lambda_ = 0.5

	epochsn = 200000
	epochs = np.arange(epochsn)
	
	model = Linear3D(4, 40, epochsn, problem='inverse', exact=exact)
	model.solve()

	fig, ax = plt.subplots(2, 1, figsize=(10, 5))

	ax[0,0].plot(epochs_list, lambdas_mu_fixed, alpha=0.5)
	ax[0,0].hlines(y=lambda_, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='Exact $\mu$')
	ax[0,0].set_xlabel("Epochs")
	ax[0,0].set_title("$\mu$ fixed with stress data")
	ax[0,0].grid()
	ax[0,0].legend()

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

if __name__ == '__main__':
	create_elasticity_pinn_data()
	#plot_inverse_elasticity()
	#plot_inverse_fixed()







