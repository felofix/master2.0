import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy

class Heart_model:
	"""
	A class for importing aand modidyfing a heart model.
	Only works with the current vtk configuration. 
	"""

	def __init__(self, filename):
		# Information about model.
		self.bc = {}
		self.bi = {}
		self.bn = {}

		self.load_file(filename)
		self.create_boundaries()

	def load_file(self, filename):
		"""
		Loads file specified when creating heart model object. 
		For now it just loads coordinates, I guess I need the 
		other stuff for fiber directin and such. 
		"""

		# For Pinns.
		self.coordinates = np.loadtxt(filename + "lv_vertices.txt")

		# Boundary idc.
		self.endo_idc = np.loadtxt(filename + "lv_endo_indeces.txt", dtype=int)
		self.epi_idc = np.loadtxt(filename + "lv_epi_indeces.txt", dtype=int)
		self.base_idc = np.loadtxt(filename + "lv_base_indeces.txt", dtype=int)

		# Boundary normals. 
		self.endo_normals = np.loadtxt(filename + "lv_endo_normals.txt")
		self.epi_normals = np.loadtxt(filename + "lv_epi_normals.txt")
		self.base_normals = np.loadtxt(filename + "lv_base_normals.txt")

	def create_boundaries(self):
		"""
		Creates the heart boundaries. 
		"""

		# Fill dictionaries. 
		# Coordinates. 
		self.bc['endo'] = self.coordinates[self.endo_idc]
		self.bc['epi'] = self.coordinates[self.epi_idc]
		self.bc['base'] = self.coordinates[self.base_idc]

		# Indeces. 
		self.bi['endo'] = self.endo_idc
		self.bi['epi'] = self.epi_idc
		self.bi['base'] = self.base_idc

		# Normals. 
		self.bn['endo'] = self.endo_normals
		self.bn['epi'] = self.epi_normals
		self.bn['base'] = self.base_normals

	def visualize_geometry(self):
		"""
		Visuaalizes geomtery. 
		Probably a nicer way to visualize this.
		"""

		x, y, z = self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		scatter = ax.scatter(x, y, z)

		# Labeling
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')
		ax.set_title('3D Scatter Plot with Function Values')

		plt.show()

	def visualize_boundaries(self):
		"""
		Visuaalizes geomtery. 
		Probably a nicer way to visualize this.
		"""

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# Plotting. 
		ax.scatter(self.bc['endo'][:, 0], self.bc['endo'][:, 1], self.bc['endo'][:, 2], color='green', label='Endo')
		ax.scatter(self.bc['epi'][:, 0], self.bc['epi'][:, 1], self.bc['epi'][:, 2], color='red', label='Epi')
		ax.scatter(self.bc['base'][:, 0], self.bc['base'][:, 1], self.bc['base'][:, 2], color='blue', label='Base')

		# Labeling
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')
		plt.legend()

		plt.show()

	def visualize_normals(self):
		"""
		Visuaalizes geomtery. 
		Probably a nicer way to visualize this.
		"""

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# Plotting. 
		ax.quiver(self.bc['endo'][:, 0], self.bc['endo'][:, 1], self.bc['endo'][:, 2],\
				   self.bn['endo'][:, 0], self.bn['endo'][:, 1], self.bn['endo'][:, 2], length = 1, color='red', label='Endo')
		ax.quiver(self.bc['epi'][:, 0], self.bc['epi'][:, 1], self.bc['epi'][:, 2],\
				   self.bn['epi'][:, 0], self.bn['epi'][:, 1], self.bn['epi'][:, 2], length = 1,color='green', label='Epi')

		# Labeling
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')

		plt.legend()
		plt.show()

