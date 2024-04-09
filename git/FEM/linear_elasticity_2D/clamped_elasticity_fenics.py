import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

# Constants
length = 1
width = 1
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

class Top(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[1], width)

class Right(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[0], length)

class Bottom(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[1], 0)

class Obstacle(SubDomain):
	def inside(self, x, on_boundary):
		return (between(x[1], (0.0, 0.1)) and between(x[0], (0.0, 1.0)))

def solve_clamped_beam_fenics(nx, ny, k):
	"""
	A two-dimensional clamped beam problem.
	As of now it is a pretty simple solver.
	"""
	# Boundaries and domain.
	left = Left()
	top = Top()
	right = Right()
	bottom = Bottom()
	obstacle = Obstacle()

	# Mesh and Vector Function Space
	mesh = fe.UnitSquareMesh(nx, ny)

	# Initialize mesh function for interior domains
	domains = MeshFunction("size_t", mesh, mesh.topology().dim())
	domains.set_all(0)
	obstacle.mark(domains, 1)

	boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	boundaries.set_all(0)

	left.mark(boundaries, 1)
	top.mark(boundaries, 2)
	right.mark(boundaries, 3)
	bottom.mark(boundaries, 4)

	# Get the coordinates of the vertices before deformation.
	coordinates_before = mesh.coordinates().copy()

	# Creating vector space for basis functions.
	V = fe.VectorFunctionSpace(mesh, "Lagrange", 1)

	# Dirichlet bouondaries. 
	dirichlet_left = fe.DirichletBC(V, fe.Constant((0, 0)), boundaries, 1)
	dirichlet_right = fe.DirichletBC(V, fe.Constant((0, 0)),boundaries, 3)
	dirichlet_bcs = [dirichlet_left, dirichlet_right]


	# Neumann.
	neumann_top = fe.Expression(('mu*pi*k*cos(k*pi*x[0])*cos(k*pi*x[1])',\
								'-2*pi*k*mu*sin(pi*k*x[0])*sin(pi*k*x[1]) + lambda_*(-pi*k*sin(pi*k*x[0])*sin(pi*k*x[1]) + pi*k*cos(pi*k*x[0]))') ,\
								pi=np.pi,k=k,mu=mu,lambda_=lambda_, degree=1)

	neumann_bottom = fe.Expression(('-mu*pi*k*cos(k*pi*x[0])*cos(k*pi*x[1])',\
								'2*pi*k*mu*sin(pi*k*x[0])*sin(pi*k*x[1]) - lambda_*(-pi*k*sin(pi*k*x[0])*sin(pi*k*x[1]) + pi*k*cos(pi*k*x[0]))') ,\
								pi=np.pi,k=k,mu=mu,lambda_=lambda_, degree=1)

	# Define new measures associated with the interior domains and
	# exterior boundaries
	dx = Measure('dx', domain=mesh, subdomain_data=domains)
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	# Define variational problem
	u = fe.TrialFunction(V)
	v = fe.TestFunction(V)
	f_x = "2*pi*pi*k*k*mu*sin(pi*k*x[0]) + pi*pi*k*k*mu*sin(pi*k*x[1])*cos(pi*k*x[0]) - lambda_*(-pi*pi*k*k*sin(pi*k*x[0]) - pi*pi*k*k*sin(pi*k*x[1])*cos(pi*k*x[0]))"
	f_y = "pi*pi*k*k*lambda_*sin(pi*k*x[0])*cos(pi*k*x[1]) + 3*pi*pi*k*k*mu*sin(pi*k*x[0])*cos(pi*k*x[1])"
	f = fe.Expression((f_x, f_y), pi=np.pi, k=k, mu=mu, lambda_=lambda_, degree=1)
	a = fe.inner(sigma(u), epsilon(v)) * fe.dx
	L = fe.dot(f, v) * dx + fe.dot(neumann_top, v) * ds(2) + fe.dot(neumann_bottom, v) * ds(4)

	# Compute solution
	u = fe.Function(V)  # This is the displacement field, that is how much it moves.
	fe.solve(a == L, u, dirichlet_bcs)

	S = TensorFunctionSpace(mesh, "Lagrange", 2)  # Tensor function space for sigma
	sigma_u = project(sigma(u), S)
	S_scalar = FunctionSpace(mesh, "Lagrange", 1)  # Scalar function space
	s_xx = project(sigma_u[0, 0], S_scalar)
	s_yy = project(sigma_u[1, 1], S_scalar)
	s_xy = project(sigma_u[0, 1], S_scalar)

	s_xx_array = []
	s_yy_array = []
	s_xy_array = []

	for vertex in vertices(mesh):
		point = vertex.point()
		s_xx_array.append(s_xx(point))
		s_yy_array.append(s_yy(point))
		s_xy_array.append(s_xy(point))

	# Move the mesh according to the displacement field u
	fe.ALE.move(mesh, u)

	# Now mesh.coordinates() will return the new coordinates of the mesh vertices
	coordinates_after = mesh.coordinates()

	return coordinates_before, coordinates_after, coordinates_after - coordinates_before, s_xx_array, s_yy_array, s_xy_array

def u_exact(x, y, k, mu, lambd):
	# Constans. 
	pi = np.pi
	cos = np.cos
	sin = np.sin

	u_x = np.sin(k*np.pi*x)
	u_y = np.sin(k*np.pi*x)*np.cos(k*np.pi*y)

	s_xx = 2*pi*k*mu*cos(pi*k*x) + lambd*(-pi*k*sin(pi*k*x)*sin(pi*k*y) + pi*k*cos(pi*k*x))
	s_yy = -2*pi*k*mu*sin(pi*k*x)*sin(pi*k*y) + lambd*(-pi*k*sin(pi*k*x)*sin(pi*k*y) + pi*k*cos(pi*k*x))
	s_xy = 1.0*pi*k*mu*cos(pi*k*x)*cos(pi*k*y)

	exact = {"u_x": u_x, "u_y": u_y, "s_xx": s_xx, "s_yy": s_yy, "s_xy": s_xy}

	
	return exact

