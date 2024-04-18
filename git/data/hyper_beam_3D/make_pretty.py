import numpy as np
import matplotlib.pyplot as plt
from dolfin import *

pinns = np.loadtxt('hyper_data_pinns.txt', delimiter=',')

fenics = np.loadtxt("neo_hookian_3d_fenics.txt", delimiter=',')

exact = {"x": fenics[:, 0].reshape(-1, 1), "y": fenics[:, 1].reshape(-1, 1), "z": fenics[:, 2].reshape(-1, 1),\
         "u_x": fenics[:, 3].reshape(-1, 1), "u_y": fenics[:, 4].reshape(-1, 1), "u_z": fenics[:, 5].reshape(-1, 1),\
         "P_xx": fenics[:, 6].reshape(-1, 1), "P_yy": fenics[:, 7].reshape(-1, 1), "P_zz": fenics[:, 8].reshape(-1, 1),\
         "P_xy": fenics[:, 9].reshape(-1, 1), "P_yx": fenics[:, 10].reshape(-1, 1), "P_xz": fenics[:, 11].reshape(-1, 1), \
         "P_zx": fenics[:, 12].reshape(-1, 1), "P_yz": fenics[:, 13].reshape(-1, 1), "P_zy": fenics[:, 14].reshape(-1, 1)}


X, Y, Z, ux, uy, uz, absz = pinns[:, 0], pinns[:, 1], pinns[:, 2], pinns[:, 3], pinns[:, 4], pinns[:, 5], pinns[:, 6]

mesh = UnitCubeMesh(9, 9, 9)

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
