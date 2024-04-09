import sys
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('../data/heart_data/linear_elasticity_data.txt', delimiter=',')
sys.path.append('../heart_model')
import heart_model as hm
sys.path.append('../pinns/linear_heart')
import linear_elasticity as le
np.random.seed(seed=1234)

def inverse_heart_percentages():
	"""Hmmm...
	"""
	mu = 1
	lambda_ = 0.5

	percentages = [0.95, 0.7, 0.5, 0.3, 0]
	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	exact = {"x": data[:, 0].reshape(-1, 1), "y": data[:, 1].reshape(-1, 1), "z": data[:, 2].reshape(-1, 1),\
			 "u_x": data[:, 3].reshape(-1, 1), "u_y": data[:, 4].reshape(-1, 1), "u_z": data[:, 5].reshape(-1, 1),\
			 "s_xx": data[:, 6].reshape(-1, 1), "s_yy": data[:, 7].reshape(-1, 1), "s_zz": data[:, 8].reshape(-1, 1),\
			 "s_xy": data[:, 9].reshape(-1, 1), "s_xz": data[:, 10].reshape(-1, 1), "s_yz": data[:, 11].reshape(-1, 1)}
	
	epochs = 50000
	epochs_list = np.arange(epochs)
	mus = np.zeros((len(percentages), epochs))
	lambdas = np.zeros((len(percentages), epochs))

	
	for i in range(len(percentages)):
		newexact = add_noise(exact.copy(), percentages[i])
		pinn = le.Linear(heart, 4, 40, epochs, problem='inverse', exact=newexact)
		pinn.solve()
		mus[i] = pinn.mus
		lambdas[i] = pinn.lambdas

	print(f'Using 100% of the data the mu relative error is: {(abs(mus[4][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the data the mu relative error is: {(abs(mus[3][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the data the mu relative error is: {(abs(mus[2][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the data the mu relative error is: {(abs(mus[1][-1] - mu)/mu*100):.2f} %')
	print(f'Using 100% of the data the mu relative error is: {(abs(mus[0][-1] - mu)/mu*100):.2f} %')

	print(f'Using 100% of the data the lambda relative error is: {(abs(lambdas[4][-1] - lambda_)/lambda_*100):.2f} %')
	print(f'Using 100% of the data the lambda relative error is: {(abs(lambdas[3][-1] - lambda_)/lambda_*100):.2f} %')
	print(f'Using 100% of the data the lambda relative error is: {(abs(lambdas[2][-1] - lambda_)/lambda_*100):.2f} %')
	print(f'Using 100% of the data the lambda relative error is: {(abs(lambdas[1][-1] - lambda_)/lambda_*100):.2f} %')
	print(f'Using 100% of the data the lambda relative error is: {(abs(lambdas[0][-1] - lambda_)/lambda_*100):.2f} %')

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=lambda_, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\lambda*$')
	colors = plt.cm.viridis(np.linspace(0, 1, len(percentages)))

	for s in range(len(percentages)):	
		ax.plot(epochs_list, mus[s], color=colors[s], alpha=0.5, label=f'{round((1 - percentages[s])*100)} % of data')
		ax.plot(epochs_list, lambdas[s], color=colors[s], alpha=0.5)
	
	ax.legend()
	plt.savefig("plots_heart/inverse_percentages_no_noise.pdf")

def inverse_heart_noise():
	mu = 1
	lambda_ = 0.5

	noises = [0.95, 0.7, 0.5, 0.3, 0]
	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	exact = {"x": data[:, 0].reshape(-1, 1), "y": data[:, 1].reshape(-1, 1), "z": data[:, 2].reshape(-1, 1),\
			 "u_x": data[:, 3].reshape(-1, 1), "u_y": data[:, 4].reshape(-1, 1), "u_z": data[:, 5].reshape(-1, 1),\
			 "s_xx": data[:, 6].reshape(-1, 1), "s_yy": data[:, 7].reshape(-1, 1), "s_zz": data[:, 8].reshape(-1, 1),\
			 "s_xy": data[:, 9].reshape(-1, 1), "s_xz": data[:, 10].reshape(-1, 1), "s_yz": data[:, 11].reshape(-1, 1)}
	
	epochs = 50000
	epochs_list = np.arange(epochs)
	mus = np.zeros((len(noises), epochs))
	lambdas = np.zeros((len(noises), epochs))
	exact = add_noise(exact.copy(), noises[0])

	for i in range(len(noises)):
		newexact = remove_percentage(exact.copy(), noises[i])
		pinn = le.Linear(heart, 4, 40, epochs, problem='inverse', exact=newexact)
		pinn.solve()
		mus[i] = pinn.mus
		lambdas[i] = pinn.lambdas

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

	fig, ax = plt.subplots()
	ax.set_title("Inverse parameters calculated by PINN")
	ax.hlines(y=mu, xmin=0, xmax=epochs, linewidth=1, color='black', linestyle='dashed', label='$\mu*$')
	ax.hlines(y=lambda_, xmin=0, xmax=epochs, linewidth=1, color='dimgray', linestyle='dashed', label='$\lambda*$')
	colors = plt.cm.viridis(np.linspace(0, 1, len(noises)))

	for s in range(len(noises)):	
		ax.plot(epochs_list, mus[s], color=colors[s], alpha=0.5, label=f'{round((1 - noises[s])*100)} % of noise')
		ax.plot(epochs_list, lambdas[s], color=colors[s], alpha=0.5)
	
	ax.legend()
	plt.savefig("plots_heart/noise.pdf")

def visualize():
	heart = hm.Heart_model("/Users/Felix/desktop/MASTER/git/master/heart_simulation/model/meshes/")
	heart.visualize_boundaries()

	
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

	# Changing exact.
	exact["x"] = exact["x"][new_indeces]
	exact["y"] = exact["y"][new_indeces]
	exact["z"] = exact["z"][new_indeces]

	exact["u_x"] = exact["u_x"][new_indeces]
	exact["u_y"] = exact["u_y"][new_indeces]
	exact["u_z"] = exact["u_z"][new_indeces]

	exact["s_xx"] = exact["s_xx"][new_indeces]
	exact["s_yy"] = exact["s_yy"][new_indeces]
	exact["s_zz"] = exact["s_zz"][new_indeces]
	exact["s_xy"] = exact["s_xy"][new_indeces]
	exact["s_xz"] = exact["s_xz"][new_indeces]
	exact["s_yz"] = exact["s_yz"][new_indeces]

	return exact

def add_noise(exact, percentage):
	max_u_x = np.max(exact["u_x"])
	max_u_y = np.max(exact["u_y"])
	max_u_z = np.max(exact["u_z"])

	max_s_xx = np.max(exact["s_xx"])
	max_s_yy = np.max(exact["s_yy"])
	max_s_zz = np.max(exact["s_zz"])
	max_s_xy = np.max(exact["s_xy"])
	max_s_xz = np.max(exact["s_xz"])
	max_s_yz = np.max(exact["s_yz"])

	noise_u_x = (np.random.normal(0, 1, len(exact["u_x"]))*max_u_x).reshape(-1, 1)
	noise_u_y = (np.random.normal(0, 1, len(exact["u_y"]))*max_u_y).reshape(-1, 1)
	noise_u_z = (np.random.normal(0, 1, len(exact["u_z"]))*max_u_z).reshape(-1, 1)

	noise_s_xx = (np.random.normal(0, 1, len(exact["s_xx"]))*max_s_xx).reshape(-1, 1)
	noise_s_yy = (np.random.normal(0, 1, len(exact["s_yy"]))*max_s_yy).reshape(-1, 1)
	noise_s_zz = (np.random.normal(0, 1, len(exact["s_zz"]))*max_s_zz).reshape(-1, 1)
	noise_s_xy = (np.random.normal(0, 1, len(exact["s_xy"]))*max_s_xy).reshape(-1, 1)
	noise_s_xz = (np.random.normal(0, 1, len(exact["s_xz"]))*max_s_xz).reshape(-1, 1)
	noise_s_yz = (np.random.normal(0, 1, len(exact["s_yz"]))*max_s_yz).reshape(-1, 1)

	exact["u_x"] += noise_u_x
	exact["u_y"] += noise_u_y
	exact["u_z"] += noise_u_z

	exact["s_xx"] += noise_s_xx
	exact["s_yy"] += noise_s_yy
	exact["s_zz"] += noise_s_zz
	exact["s_xy"] += noise_s_xy
	exact["s_xz"] += noise_s_xz
	exact["s_yz"] += noise_s_yz

	return exact


if __name__ == '__main__':
	#inverse_heart_percentages()
	#inverse_heart_noise()
	visualize()





