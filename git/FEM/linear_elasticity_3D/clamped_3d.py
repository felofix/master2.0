import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

# Constants
mu = 1.0
lambda_ = 0.5

def epsilon(u):
	# Engineering strain.
	return 0.5 * (fe.nabla_grad(u) + fe.nabla_grad(u).T)

def sigma(u):    # Stress.
	return lambda_ * fe.div(u) * fe.Identity(u.geometric_dimension()) + 2 * mu * epsilon(u)

	# Boundaries. 
class Left(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[0], 0)

class Right(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[0], 1)

class Bottom(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[2], 0)

class Top(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[2], 1)

class Front(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[1], 0)

class Back(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[1], 1)

def solve_clamped_beam_fenics(nx, ny, nz):
	"""
	A two-dimensional clamped beam problem.
	As of now it is a pretty simple solver.
	"""
	# Boundaries and domain.
	left = Left()
	top = Top()
	right = Right()

	# Mesh and Vector Function Space
	mesh = fe.UnitCubeMesh(nx, ny, nz)

	boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	boundaries.set_all(0)

	left.mark(boundaries, 1)
	top.mark(boundaries, 2)
	right.mark(boundaries, 3)

	# Get the coordinates of the vertices before deformation.
	coordinates_before = mesh.coordinates().copy()

	# Creating vector space for basis functions.
	V = fe.VectorFunctionSpace(mesh, "Lagrange", 1)

	# Dirichlet bouondaries. 
	dirichlet_left  = fe.DirichletBC(V, fe.Constant((0, 0, 0)), boundaries, 1)
	dirichlet_bcs = [dirichlet_left] #, dirichlet_right]

	# Neumann.
	neumann_top = fe.Constant((0, 0, 0))

	# Define new measures associated with the interior domains and
	# exterior boundaries
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
	f = fe.Constant((0, 0, -0.2))

	# Define variational problem
	u = fe.TrialFunction(V)
	v = fe.TestFunction(V)
	a = fe.inner(sigma(u), epsilon(v)) * fe.dx
	L = fe.dot(f, v)*fe.dx + fe.dot(neumann_top, v) * ds(2)

	# Compute solution
	u = fe.Function(V)  # This is the displacement field, that is how much it moves.
	fe.solve(a == L, u, dirichlet_bcs)

	utest = grad(u)
	S = TensorFunctionSpace(mesh, "Lagrange", 1)  # Tensor function space for sigma
	sigma_u = project(sigma(u), S)
	S_scalar = FunctionSpace(mesh, "Lagrange", 1)  # Scalar function space

	u_xx = project(utest[0, 0], S_scalar)
	u_yy = project(utest[1, 1], S_scalar)
	u_zz = project(utest[2, 2], S_scalar)
	u_xy = project(utest[0, 1], S_scalar)
	u_yx = project(utest[1, 0], S_scalar)
	u_xz = project(utest[0, 2], S_scalar)
	u_zx = project(utest[2, 0], S_scalar)
	u_yz = project(utest[1, 2], S_scalar)
	u_zy = project(utest[2, 1], S_scalar)

	s_xx = project(sigma_u[0, 0], S_scalar)
	s_yy = project(sigma_u[1, 1], S_scalar)
	s_zz = project(sigma_u[2, 2], S_scalar)
	s_xy = project(sigma_u[0, 1], S_scalar)
	s_xz = project(sigma_u[0, 2], S_scalar)
	s_yz = project(sigma_u[1, 2], S_scalar)

	s_xx_array = []
	s_yy_array = []
	s_zz_array = []
	s_xy_array = []
	s_xz_array = []
	s_yz_array = []

	u_xx_array = []
	u_yy_array = []
	u_zz_array = []
	u_xy_array = []
	u_yx_array = []
	u_xz_array = []
	u_zx_array = []
	u_yz_array = []
	u_zy_array = []

	for vertex in vertices(mesh):
		point = vertex.point()
		s_xx_array.append(s_xx(point))
		s_yy_array.append(s_yy(point))
		s_zz_array.append(s_zz(point))
		s_xy_array.append(s_xy(point))
		s_xz_array.append(s_xz(point))
		s_yz_array.append(s_yz(point))

		u_xx_array.append(u_xx(point))
		u_yy_array.append(u_yy(point))
		u_zz_array.append(u_zz(point))
		u_xy_array.append(u_xy(point))
		u_yx_array.append(u_yx(point))
		u_xz_array.append(u_xz(point))
		u_zx_array.append(u_zx(point))
		u_yz_array.append(u_yz(point))
		u_zy_array.append(u_zy(point))

	# Move the mesh according to the displacement field u
	fe.ALE.move(mesh, u)

	# Now mesh.coordinates() will return the new coordinates of the mesh vertices
	coordinates_after = mesh.coordinates()

	return coordinates_before, coordinates_after, coordinates_after - coordinates_before, \
		   s_xx_array, s_yy_array, s_zz_array, s_xy_array, s_xz_array, s_yz_array, \
		   u_xx_array, u_yy_array, u_zz_array, u_xy_array, u_yx_array, u_xz_array, \
		   u_zx_array, u_yz_array, u_zy_array
