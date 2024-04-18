import sys
import numpy as np
sys.path.append('../pinns/hyper_elasticity_3D')
import neo_hookian_3d as h3d
import matplotlib.pyplot as plt
import torch

fenics = np.loadtxt("../data/hyper_beam_3D/neo_hookian_3d_fenics.txt", delimiter=',')

exact = {"x": fenics[:, 0].reshape(-1, 1), "y": fenics[:, 1].reshape(-1, 1), "z": fenics[:, 2].reshape(-1, 1),\
		 "u_x": fenics[:, 3].reshape(-1, 1), "u_y": fenics[:, 4].reshape(-1, 1), "u_z": fenics[:, 5].reshape(-1, 1),\
		 "P_xx": fenics[:, 6].reshape(-1, 1), "P_yy": fenics[:, 7].reshape(-1, 1), "P_zz": fenics[:, 8].reshape(-1, 1),\
		 "P_xy": fenics[:, 9].reshape(-1, 1), "P_yx": fenics[:, 10].reshape(-1, 1), "P_xz": fenics[:, 11].reshape(-1, 1), \
		 "P_zx": fenics[:, 12].reshape(-1, 1), "P_yz": fenics[:, 13].reshape(-1, 1), "P_zy": fenics[:, 14].reshape(-1, 1)}

def create_elasticity_pinn_data():
	epochsn = 200000
	epochs = np.arange(epochsn)
	model = h3d.Neo_Hookian_3D(4, 40, epochsn, problem='forward', exact=exact)
	model.solve()

	X, Y, Z = model.X, model.Y, model.Z

	u = model.predict(X, Y, Z)
	ux, uy, uz, pxx, pyy, pzz, pxy, pyx, pxz, pzx, pyz, pzy = u[:, 0].reshape(-1, 1), u[:, 1].reshape(-1, 1), u[:, 2].reshape(-1, 1), u[:, 3].reshape(-1, 1), \
										   				  	  u[:, 4].reshape(-1, 1), u[:, 5].reshape(-1, 1), u[:, 6].reshape(-1, 1), u[:, 7].reshape(-1, 1), \
										   				  	  u[:, 8].reshape(-1, 1), u[:, 9].reshape(-1, 1), u[:, 10].reshape(-1, 1), u[:, 11].reshape(-1, 1)

	with open('../data/hyper_beam_3D/hyper_data_pinns.txt', 'w') as file:
		for i in range(len(pxx)):
			file.write(f'{X[i][0]}, {Y[i][0]}, {Z[i][0]}, {ux[i][0]}, {uy[i][0]}, {uz[i][0]}, {abs(uz[i] - exact["u_z"][i])[0]} \n')

	# Printing the errors. 
	print(f'The MSE for u_x using Mixed PINNS is: {mean_squared_error(exact["u_x"], ux)}')
	print(f'The MSE for u_y using Mixed PINNS is: {mean_squared_error(exact["u_y"], uy)}')
	print(f'The MSE for u_y using Mixed PINNS is: {mean_squared_error(exact["u_z"], uz)}')

	print(f'The MSE for p_xx using Mixed PINNS is: {mean_squared_error(exact["P_xx"], pxx)}')
	print(f'The MSE for p_yy using Mixed PINNS is: {mean_squared_error(exact["P_yy"], pyy)}')
	print(f'The MSE for p_zz using Mixed PINNS is: {mean_squared_error(exact["P_zz"], pzz)}')

	print(f'The MSE for p_xy using Mixed PINNS is: {mean_squared_error(exact["P_xy"], pxy)}')
	print(f'The MSE for p_yx using Mixed PINNS is: {mean_squared_error(exact["P_yx"], pxy)}')
	print(f'The MSE for p_xz using Mixed PINNS is: {mean_squared_error(exact["P_xz"], pxz)}')
	print(f'The MSE for p_zx using Mixed PINNS is: {mean_squared_error(exact["P_zx"], pxz)}')
	print(f'The MSE for p_yz using Mixed PINNS is: {mean_squared_error(exact["P_yz"], pyz)}')
	print(f'The MSE for p_zy using Mixed PINNS is: {mean_squared_error(exact["P_zy"], pyz)}')

	plot_loss(model.losses)

def plot_loss(mses):
	epochs = np.arange(len(mses))
	plt.loglog(epochs, mses, label="Total loss", alpha=0.5)
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("MSE")
	plt.grid()
	plt.savefig("plots_hyper_3D/losses.pdf")

def plot_inverse_elasticity():
	mu = 1
	kappa = 0.5

	epochsn = 200000
	epochs = np.arange(epochsn)
	model = h3d.Neo_Hookian_3D(4, 40, epochsn, problem='inverse', exact=exact)
	model.solve()

	print(f'The last measured kappa value was: {model.kappas[-1]:.2f}')
	print(f'The last measured mu value was: {model.mus[-1]:.2f}')

	fig, ax = plt.subplots()
	ax.plot(epochs, model.kappas, alpha=0.5, label='$\lambda$')
	ax.plot(epochs, model.mus, alpha=0.5, label='$\mu$')
	ax.hlines(y= kappa, xmin=0, xmax=len(epochs), linewidth=1, color='black', linestyle='dashed', label='Exact $\kappa$')
	ax.hlines(y= mu, xmin=0, xmax=len(epochs), linewidth=1, color='black', linestyle='dashed', label='Exact $\mu$')
	ax.set_xlabel("Epochs")
	ax.set_title("$\mu$ fixed with stress data")
	ax.grid()
	ax.legend()
	plt.savefig("plots_hyper_3D/inverse.pdf")

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

if __name__ == '__main__':
	create_elasticity_pinn_data()
	#plot_inverse_elasticity()
	#plot_inverse_fixed()







