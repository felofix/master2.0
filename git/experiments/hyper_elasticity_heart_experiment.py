import sys
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('../data/heart_data/hyper_elasticity_data.txt', delimiter=',')
data_a = np.loadtxt('../data/heart_data/hyper_elasticity_data_active.txt', delimiter=',')
sys.path.append('../heart_model')
import heart_model as hm
sys.path.append('../pinns/hyper_heart')
import hyper_elasticity as he
np.random.seed(seed=1234)
import copy
import seaborn as sns

exact = {"x": data[:, 0].reshape(-1, 1), "y": data[:, 1].reshape(-1, 1), "z": data[:, 2].reshape(-1, 1),\
			 "u_x": data[:, 3].reshape(-1, 1), "u_y": data[:, 4].reshape(-1, 1), "u_z": data[:, 5].reshape(-1, 1),\
			 "P_xx": data[:, 6].reshape(-1, 1), "P_yy": data[:, 7].reshape(-1, 1), "P_zz": data[:, 8].reshape(-1, 1),\
			 "P_xy": data[:, 9].reshape(-1, 1), "P_yx": data[:, 10].reshape(-1, 1), "P_xz": data[:, 11].reshape(-1, 1), \
			 "P_zx": data[:, 12].reshape(-1, 1), "P_yz": data[:, 13].reshape(-1, 1), "P_zy": data[:, 14].reshape(-1, 1)}

exact_active = {"x": data[:, 0].reshape(-1, 1), "y": data[:, 1].reshape(-1, 1), "z": data[:, 2].reshape(-1, 1),\
				"u_x": data[:, 3].reshape(-1, 1), "u_y": data[:, 4].reshape(-1, 1), "u_z": data[:, 5].reshape(-1, 1),\
			 	"P_xx": data[:, 6].reshape(-1, 1), "P_yy": data[:, 7].reshape(-1, 1), "P_zz": data[:, 8].reshape(-1, 1),\
			 	"P_xy": data[:, 9].reshape(-1, 1), "P_yx": data[:, 10].reshape(-1, 1), "P_xz": data[:, 11].reshape(-1, 1), \
				"P_zx": data[:, 12].reshape(-1, 1), "P_yz": data[:, 13].reshape(-1, 1), "P_zy": data[:, 14].reshape(-1, 1), \
				"f0_x": data_a[:, 15].reshape(-1, 1), "f0_y": data_a[:, 16].reshape(-1, 1), "f0_z": data_a[:, 17].reshape(-1, 1)}

def inverse_heart():
	mu = 1
	kappa = 0.5

	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")

	epochs = 50000
	epochs_list = np.arange(epochs)
	seeds = [123, 500, 1000, 2024]

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN by using stress data for the Neo Hookian model")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=kappa, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\kappa*$')
	colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))

	for s in range(len(seeds)):
		pinn = he.Neo_Hookian(heart, 4, 40, epochs, problem='inverse', exact=exact, seed = seeds[s])
		pinn.solve()	
		ax.plot(epochs_list, pinn.mus, color=colors[s],alpha=0.5)
		ax.plot(epochs_list, pinn.kappas, color=colors[s],alpha=0.5)
		print(f'The last mu calculated was: {pinn.mus[-1]:.3f}')
		print(f'The last kappa calculated was: {pinn.kappas[-1]:.3f}')

	ax.legend()
	plt.grid()
	plt.savefig("plots_heart_hyper/inverse.pdf")

def inverse_heart_percentages():
	"""Hmmm...
	"""
	mu = 1
	kappa = 0.5

	percentages = [0.95, 0.7, 0.5, 0.3]
	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	
	epochs = 20000
	epochs_list = np.arange(epochs)
	mus = np.zeros((len(percentages), epochs))
	kappas = np.zeros((len(percentages), epochs))

	
	for i in range(len(percentages)):
		newexact = remove_percentage(exact.copy(), percentages[i])
		pinn = he.Neo_Hookian(heart, 4, 40, epochs, problem='inverse', exact=newexact)
		pinn.solve()
		mus[i] = pinn.mus
		kappas[i] = pinn.kappas

	print(f'Using 100% of the data the mu relative error is: {(abs(mus[3][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the data the mu relative error is: {(abs(mus[2][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the data the mu relative error is: {(abs(mus[1][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the data the mu relative error is: {(abs(mus[0][-1] - mu)/mu*100):.2f} %')

	print(f'Using 100% of the data the lambda relative error is: {(abs(kappas[3][-1] - kappa)/kappa*100):.2f} %')
	print(f'Using 100% of the data the lambda relative error is: {(abs(kappas[2][-1] - kappa)/kappa*100):.2f} %')
	print(f'Using 100% of the data the lambda relative error is: {(abs(kappas[1][-1] - kappa)/kappa*100):.2f} %')
	print(f'Using 100% of the data the lambda relative error is: {(abs(kappas[0][-1] - kappa)/kappa*100):.2f} %')

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=kappa, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\kappa*$')
	colors = plt.cm.viridis(np.linspace(0, 1, len(percentages)))

	for s in range(len(percentages)):	
		ax.plot(epochs_list, mus[s], color=colors[s], alpha=0.5, label=f'{round((1 - percentages[s])*100)} % of data')
		ax.plot(epochs_list, kappas[s], color=colors[s], alpha=0.5)
	
	ax.legend()
	plt.grid()
	plt.savefig("plots_heart_hyper/inverse_percentages_no_noise.pdf")

def inverse_heart_noise():
	mu = 1
	kappa = 0.5

	noises = [0.1, 0.05,0.03, 0.01]

	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	
	epochs = 20000
	epochs_list = np.arange(epochs)

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=kappa, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\kappa*$')
	colors = plt.cm.viridis(np.linspace(0, 1, len(noises)))

	for s in range(len(noises)):
		newexact = add_noise(exact.copy(), noises[s])
		pinn = he.Neo_Hookian(heart, 4, 40, epochs, problem='inverse', exact=newexact)
		pinn.solve()
		ax.plot(epochs_list, pinn.mus, color=colors[s], alpha=0.5, label=f'{noises[s]*100} % of noise')
		ax.plot(epochs_list, pinn.kappas, color=colors[s], alpha=0.5)

	"""
	print(f'Using 0% of the noise the mu relative error is: {(abs(mus[4][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the noise the mu relative error is: {(abs(mus[3][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the noise the mu relative error is: {(abs(mus[2][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the noise the mu relative error is: {(abs(mus[1][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the noise the mu relative error is: {(abs(mus[0][-1] - mu)/mu*100):.2f} %')

	print(f'Using 100% of the noise the lambda relative error is: {(abs(lambdas[4][-1] - lambda_)/lambda_*100):.2f} %')
	print(f'Using 100% of the noise the lambda relative error is: {(abs(lambdas[3][-1] - lambda_)/lambda_*100):.2f} %')
	print(f'Using 100% of the noise the lambda relative error is: {(abs(lambdas[2][-1] - lambda_)/lambda_*100):.2f} %')
	print(f'Using 100% of the noise the lambda relative error is: {(abs(lambdas[1][-1] - lambda_)/lambda_*100):.2f} %')
	print(f'Using 100% of the noise the lambda relative error is: {(abs(lambdas[0][-1] - lambda_)/lambda_*100):.2f} %')
	"""
	
	ax.legend()
	plt.grid()
	plt.savefig("plots_heart_hyper/noise.pdf")

def visualize():
	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	heart.visualize_boundaries()

def inverse_active_model():
	mu = 1
	kappa = 0.5
	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	epochs = 50000

	percentages = [0.95, 0.7, 0.5, 0.3]
	noises = [0.1, 0.05,0.03, 0.01]

	mistakes = np.zeros((len(percentages), len(noises)))

	for p in range(len(percentages)):
		for n in range(len(noises)):
			newexact = remove_percentage(exact_active.copy(), percentages[p])
			newexact = add_noise(newexact.copy(), noises[n])
			pinn = he.Neo_Hookian(heart, 4, 40, epochs, problem='inverse', exact=newexact)
			pinn.solve()
			last_mu = pinn.mus[-1]
			last_kappa = pinn.kappas[-1]
			total_mistake = abs(last_mu - mu) + abs(last_kappa - kappa)
			mistakes[p, n] = total_mistake

	plot_heatmap(mistakes, percentages, noises, "Total deviation from $\mu$ and $\kappa$")


def inverse_active_long():	
	mu = 1
	kappa = 0.5
	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	epochs = 1000000
	epochs_list = np.arange(epochs)
	newexact = remove_percentage(exact_active.copy(), 0.95)
	newexact = add_noise(newexact.copy(), 0.1)
	pinn = he.Neo_Hookian(heart, 4, 40, epochs, problem='inverse', exact=newexact)
	pinn.solve()

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN by using stress data for the Neo Hookian model")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=kappa, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\kappa*$')

	ax.plot(epochs_list, pinn.mus,  alpha=0.5)
	ax.plot(epochs_list, pinn.kappas,  alpha=0.5)

	print(f'The last calculated mu was: {pinn.mus[-1]}')
	print(f'The last calculated kappa was: {pinn.kappas[-1]}')
	
	ax.legend()
	plt.grid()
	plt.savefig("plots_heart_hyper/inverse_active_long.pdf")

def create_heart_points_plot():
	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	newexact = remove_percentage(exact_active.copy(), 0.95)
	newexact = add_noise(newexact.copy(), 0.1)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(exact["x"], exact["y"], exact["z"], alpha=0.1, color='black', label="Before")
	ax.scatter(newexact["x"], newexact["y"], newexact["z"], color='red', label="After")
	plt.legend()
	plt.show()


def plot_heatmap(matrix, row_labels, col_labels, title, cmap="YlGnBu"):
	"""
	Plot a heatmap using Seaborn.

	Parameters:
		matrix (numpy.array): 2D array to be plotted.
		row_labels (list): Labels for the rows.
		col_labels (list): Labels for the columns.
		title (str): Title of the heatmap.
		cmap (str, optional): Color map. Defaults to "YlGnBu".

	Returns:
		None
	"""
	plt.figure(figsize=(10, 8))
	
	# Create a heatmap using Seaborn
	sns.heatmap(matrix, annot=True, cmap=cmap, cbar=True, 
				xticklabels=col_labels, yticklabels=row_labels, 
				fmt=".2e", linewidths=0.5)

	# Setting the title
	plt.title(title, fontsize=18)
	
	# Setting x and y labels
	plt.xlabel('Noise values', fontsize=14)
	plt.ylabel('Percentage values', fontsize=14)

	plt.savefig("plots_heart_hyper/" + title)

	
def show_heart_before_and_after(x, y, z, ux, uy, uz):
	"""
	Shows the heart before and after. 
	"""
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(x, y, z, color='black', label='before')
	ax.scatter(x + ux, y + uy, z + uz, color='red', label='after')
	plt.legend()
	plt.show()

def heart_and_displacement(x, y, z, ux, uy, uz):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	#ax.scatter(x, y, z, color='blue', label='heart')
	ax.quiver(x, y, z, ux, uy, uz, color='red', alpha=0.2, label='disp')
	plt.legend()
	plt.show()

def show_heart_stress(x, y, z, stress):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	scatter = ax.scatter(x, y, z, c = stress)
	plt.colorbar(scatter)
	plt.show()

def mean_squared_error(y_true, y_pred):
	return np.mean((y_true - y_pred) ** 2)

def remove_percentage(exact, percentage):
	indeces = np.arange(len(exact["x"]))

	total_elements_to_remove = int(len(indeces) * percentage)
	indices_to_remove = np.random.choice(len(indeces), total_elements_to_remove, replace=False)
	new_indeces = np.delete(indeces, indices_to_remove)

	# Changing coordinates. 
	exact["x"] = exact["x"][new_indeces]
	exact["y"] = exact["y"][new_indeces]
	exact["z"] = exact["z"][new_indeces]

	# Changing exact.
	exact["u_x"] = exact["u_x"][new_indeces]
	exact["u_y"] = exact["u_y"][new_indeces]
	exact["u_z"] = exact["u_z"][new_indeces]

	exact["P_xx"] = exact["P_xx"][new_indeces]
	exact["P_yy"] = exact["P_yy"][new_indeces]
	exact["P_zz"] = exact["P_zz"][new_indeces]
	exact["P_xy"] = exact["P_xy"][new_indeces]
	exact["P_yx"] = exact["P_yx"][new_indeces]
	exact["P_xz"] = exact["P_xz"][new_indeces]
	exact["P_zx"] = exact["P_zx"][new_indeces]
	exact["P_yz"] = exact["P_yz"][new_indeces]
	exact["P_zy"] = exact["P_zy"][new_indeces]

	return exact

def add_noise(exact, percentage):
	exacter = copy.deepcopy(exact)
	max_u_x = np.max(exacter["u_x"])
	max_u_y = np.max(exacter["u_y"])
	max_u_z = np.max(exacter["u_z"])

	max_P_xx = np.max(exacter["P_xx"])
	max_P_yy = np.max(exacter["P_yy"])
	max_P_zz = np.max(exacter["P_zz"])
	max_P_xy = np.max(exacter["P_xy"])
	max_P_yx = np.max(exacter["P_yx"])
	max_P_xz = np.max(exacter["P_xz"])
	max_P_zx = np.max(exacter["P_zx"])
	max_P_yz = np.max(exacter["P_yz"])
	max_P_zy = np.max(exacter["P_zy"])

	noise_u_x = (np.random.normal(0, 1, len(exacter["u_x"]))*max_u_x*percentage).reshape(-1, 1)
	noise_u_y = (np.random.normal(0, 1, len(exacter["u_y"]))*max_u_y*percentage).reshape(-1, 1)
	noise_u_z = (np.random.normal(0, 1, len(exacter["u_z"]))*max_u_z*percentage).reshape(-1, 1)

	noise_P_xx = (np.random.normal(0, 1, len(exacter["P_xx"]))*max_P_xx*percentage).reshape(-1, 1)
	noise_P_yy = (np.random.normal(0, 1, len(exacter["P_yy"]))*max_P_yy*percentage).reshape(-1, 1)
	noise_P_zz = (np.random.normal(0, 1, len(exacter["P_zz"]))*max_P_zz*percentage).reshape(-1, 1)
	noise_P_xy = (np.random.normal(0, 1, len(exacter["P_xy"]))*max_P_xy*percentage).reshape(-1, 1)
	noise_P_yx = (np.random.normal(0, 1, len(exacter["P_yx"]))*max_P_yx*percentage).reshape(-1, 1)
	noise_P_xz = (np.random.normal(0, 1, len(exacter["P_xz"]))*max_P_xz*percentage).reshape(-1, 1)
	noise_P_zx = (np.random.normal(0, 1, len(exacter["P_zx"]))*max_P_zx*percentage).reshape(-1, 1)
	noise_P_yz = (np.random.normal(0, 1, len(exacter["P_yz"]))*max_P_yz*percentage).reshape(-1, 1)
	noise_P_zy = (np.random.normal(0, 1, len(exacter["P_zy"]))*max_P_zy*percentage).reshape(-1, 1)

	exacter["u_x"] += noise_u_x
	exacter["u_y"] += noise_u_y
	exacter["u_z"] += noise_u_z

	exacter["P_xx"] += noise_P_xx
	exacter["P_yy"] += noise_P_yy
	exacter["P_zz"] += noise_P_zz
	exacter["P_xy"] += noise_P_xy
	exacter["P_yx"] += noise_P_yx
	exacter["P_xz"] += noise_P_xz
	exacter["P_zx"] += noise_P_zx
	exacter["P_yz"] += noise_P_yz
	exacter["P_zy"] += noise_P_zy

	return exacter


if __name__ == '__main__':
	#inverse_heart()
	#inverse_heart_percentages()
	#inverse_heart_noise()
	#visualize()
	#inverse_active_model()
	#inverse_active_long()
	create_heart_points_plot()





