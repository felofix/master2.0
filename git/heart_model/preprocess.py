import numpy as np
import cardiac_geometries
from dolfin import *
import matplotlib.pyplot as plt
import h5py

class Preprocess:
	def __init__(self, filename):
		"""	
		I'm just going to try to remoove all points on the boundary so that I only need to trhink about the centroids oon the boundary. 
		Maybe not gonna work, ich weist nicht. 
		"""
		self.filename = filename
		geo = cardiac_geometries.geometry.Geometry.from_folder(filename)
		ffun = geo.ffun

		all_vertices = 0

		with h5py.File(filename + '/vertex_mesh.h5', 'r') as file:
			# List all groups
			all_vertices = list(file['data0'])

		all_vertices = np.array(all_vertices)

		# Midpoints. Remember to create inidices that refer to them in the total amount of points. 
		midpoints_endo = []
		midpoints_epi = []
		midpoints_base = []

		# Vertice indicies for the boundaries. 
		vert_ind_endo = []
		vert_ind_epi = []
		vert_ind_base = []

		# Normals. 
		normal_endo = []
		normal_epi = []
		normal_base = []


		for facet in facets(geo.mesh):

			if geo.ffun[facet] == geo.markers["ENDO"][0]:
				# Mid points.
				vertices_mid = []
				for vertex in facet.midpoint():
					vertices_mid.append(vertex)

				midpoints_endo.append(vertices_mid)

				# Vertice indeces.
				for idc in facet.entities(0):
					vert_ind_endo.append(idc)

				# Normal.
				normal_endo.append(facet.normal().array())

			if geo.ffun[facet] == geo.markers["EPI"][0]:
				# Mid points.
				vertices_mid = []
				for vertex in facet.midpoint():
					vertices_mid.append(vertex)

				midpoints_epi.append(vertices_mid)

				# Vertice indeces.
				for idc in facet.entities(0):
					vert_ind_epi.append(idc)

				# Normal.
				normal_epi.append(facet.normal().array())

			if geo.ffun[facet] == geo.markers["BASE"][0]:
				# Mid points.
				vertices_mid = []
				for vertex in facet.midpoint():
					vertices_mid.append(vertex)

				midpoints_base.append(vertices_mid)

				# Vertice indeces.
				for idc in facet.entities(0):
					vert_ind_base.append(idc)

				# Normal.
				normal_base.append(facet.normal().array())

		midpoints_endo = np.array(midpoints_endo)
		midpoints_epi = np.array(midpoints_epi)
		midpoints_base = np.array(midpoints_base)

		vert_ind_endo = np.array(vert_ind_endo).squeeze()
		vert_ind_epi = np.array(vert_ind_epi).squeeze()
		vert_ind_base = np.array(vert_ind_base).squeeze()

		# Normals. 
		normal_endo = np.array(normal_endo)
		normal_epi = np.array(normal_epi)
		normal_base = np.array(normal_base)

		idc_remove = np.concatenate((vert_ind_endo, vert_ind_epi, vert_ind_base))
		idc_remove = np.unique(idc_remove)

		mask = np.ones(len(all_vertices), dtype=bool)
		mask[idc_remove] = False
		all_vertices = all_vertices[mask]

		# Adding midpoints and savnig indices. 
		len_endo = len(midpoints_endo)
		len_epi = len(midpoints_epi)
		len_base = len(midpoints_base)

		idc_endo = np.arange(len(all_vertices), len_endo + len(all_vertices))
		all_vertices = np.vstack((all_vertices, midpoints_endo))

		idc_epi = np.arange(len(all_vertices), len_epi + len(all_vertices))
		all_vertices = np.vstack((all_vertices, midpoints_epi))

		idc_base = np.arange(len(all_vertices), len_base + len(all_vertices))
		all_vertices = np.vstack((all_vertices, midpoints_base))

		# Write out the information in a meaningful way. 

		with open('meshes/lv_vertices.txt', 'w') as writer:
			for v in all_vertices:
				writer.write(f'{v[0]} {v[1]} {v[2]} \n')

		with open('meshes/lv_endo_indeces.txt', 'w') as writer:
			for ie in idc_endo:
				writer.write(f'{ie}\n')

		with open('meshes/lv_epi_indeces.txt', 'w') as writer:
			for ip in idc_epi:
				writer.write(f'{ip}\n')

		with open('meshes/lv_base_indeces.txt', 'w') as writer:
			for ib in idc_base:
				writer.write(f'{ib}\n')
			
		with open('meshes/lv_endo_normals.txt', 'w') as writer:
			for ne in normal_endo:
				writer.write(f'{ne[0]} {ne[1]} {ne[2]} \n')

		with open('meshes/lv_epi_normals.txt', 'w') as writer:
			for nep in normal_epi:
				writer.write(f'{nep[0]} {nep[1]} {nep[2]} \n')

		with open('meshes/lv_base_normals.txt', 'w') as writer:
			for nb in normal_base:
				writer.write(f'{nb[0]} {nb[1]} {nb[2]} \n')



prepro = Preprocess("meshes/lv-mesh")