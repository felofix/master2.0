import numpy as np
import matplotlib.pyplot as plt
from dolfin import *

pinns = np.loadtxt('linear_data_pinns.txt', delimiter=',')

data = np.loadtxt("linear_data_fenics.txt", delimiter=',')
exact = {"x": data[:, 0].reshape(-1, 1), "y": data[:, 1].reshape(-1, 1), "z": data[:, 2].reshape(-1, 1),\
		 "u_x": data[:, 3].reshape(-1, 1), "u_y": data[:, 4].reshape(-1, 1), "u_z": data[:, 5].reshape(-1, 1),\
		 "s_xx": data[:, 6].reshape(-1, 1), "s_yy": data[:, 7].reshape(-1, 1), "s_zz": data[:, 8].reshape(-1, 1),\
		 "s_xy": data[:, 9].reshape(-1, 1), "s_xz": data[:, 10].reshape(-1, 1), "s_yz": data[:, 11].reshape(-1, 1), \
		 "u_xx": data[:, 12].reshape(-1, 1), "u_yy": data[:, 13].reshape(-1, 1), "u_zz": data[:, 14].reshape(-1, 1),\
		 "u_xy": data[:, 15].reshape(-1, 1), "u_yx": data[:, 16].reshape(-1, 1), "u_xz": data[:, 17].reshape(-1, 1), \
		 "u_zx": data[:, 18].reshape(-1, 1), "u_yz": data[:, 19].reshape(-1, 1), "u_zy": data[:, 20].reshape(-1, 1)}


X, Y, Z, ux, uy, uz, absz = pinns[:, 0], pinns[:, 1], pinns[:, 2], pinns[:, 3], pinns[:, 4], pinns[:, 5], pinns[:, 6]

mesh = UnitCubeMesh(10, 10, 10)

V_scalar = FunctionSpace(mesh, 'Lagrange', 1)
vertex_values = Function(V_scalar)
vertex_fenics = Function(V_scalar)
vertex_diff = Function(V_scalar)

class MyExpression(UserExpression):
    def __init__(self, mesh, values, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.values = values
    def eval_cell(self, value, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        vertex_indices = cell.entities(0)
        value[0] = np.mean([self.values[vi] for vi in vertex_indices])

# Assuming uz is aligned with vertices and contains the correct values
expr = MyExpression(mesh, uz, degree=1)
expr2 = MyExpression(mesh, exact["u_z"], degree=1)
expr3 = MyExpression(mesh, absz, degree=1)

# Interpolate this expression onto your scalar function space
vertex_values.interpolate(expr)
vertex_fenics.interpolate(expr2)
vertex_diff.interpolate(expr3)

file_values = File('vertex_values.pvd')
file_values << vertex_values

file_values = File('vertex_values_fenics.pvd')
file_values << vertex_fenics

file_values = File('vertex_values_abs.pvd')
file_values << vertex_diff

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
