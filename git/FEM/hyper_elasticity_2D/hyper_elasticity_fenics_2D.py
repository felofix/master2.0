import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
import ufl

mu = 1.0
kappa = 0.5

	# Boundaries. 
class Left(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[0], 0)

class Right(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[0], 1)

class Bottom(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[1], 0)

class Top(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and fe.near(x[1], 1)

def neo_hookean(F):
	r"""Neo Hookean model

	.. math::
		\Psi(F) = \frac{\mu}{2}(I_1 - 3)

	Parameters
	----------
	F : ufl.Coefficient
		Deformation gradient
	mu : float, optional
		Material parameter, by default 15.0

	Returns
	-------
	ufl.Coefficient
		Strain energy density
	"""
	C = F.T * F
	I1 = tr(C)
	return 0.5 * mu * (I1 - 3)

def compressibility(F):
	r"""Penalty for compressibility
	.. math::
		\kappa (J \mathrm{ln}J - J + 1)
	Parameters
	----------
	F : ufl.Coefficient
		Deformation gradient
	kappa : float, optional
		Parameter for compressibility, by default 1e3
	Returns
	-------
	ufl.Coefficient
		Energy for compressibility
	"""
	J = det(F)
	return kappa/2*(ln(J))**2 - mu*ln(J)

def solve_neo_hookian(nx, ny):
	# Create a Unit Cube Mesh
	# Boundaries and domain.
	left = Left()
	right = Right()
	top = Top()
	bottom = Bottom()

	# Mesh and Vector Function Space
	mesh = fe.UnitSquareMesh(nx, ny)

	boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	boundaries.set_all(0)

	left.mark(boundaries, 1)
	right.mark(boundaries, 2)
	top.mark(boundaries, 3)
	bottom.mark(boundaries, 4)

	coordinates_before = mesh.coordinates().copy()

	# Function space for the displacement
	V = VectorFunctionSpace(mesh, "Lagrange", 1)
	# The displacement
	u = Function(V)
	# Test function for the displacement
	u_test = TestFunction(V)

	# Function space for scalar fields (to project tensor components)
	S = FunctionSpace(mesh, "Lagrange", 1)

	# Compute the deformation gradient
	F = ufl.variable(grad(u) + Identity(2))


	J = det(F)

	psi = neo_hookean(F) + compressibility(F)

	neumann_right = fe.Constant('0.3')
	N = FacetNormal(mesh)
	n = neumann_right * J*inv(F).T * N

	# Define new measures associated with the interior domains and
	# exterior boundaries
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	# We specify that the displacement should be zero in all directions
	bcs = [DirichletBC(V, Constant((0.0, 0.0)), left)]

	Pi = psi * dx

	# Compute first variation of Pi (directional derivative about u in the direction of v)
	F_variational = derivative(Pi, u, u_test) + inner(u_test, n) * ds(3)
	solve(F_variational == 0, u, bcs=bcs)

	# Compute P after solving the problem
	P = diff(psi, F)

	P_xx = project(P[0, 0], S)
	P_yy = project(P[1, 1], S)
	P_xy = project(P[0, 1], S)
	P_yx = project(P[1, 0], S)

	P_xx_array = []
	P_yy_array = []
	P_xy_array = []
	P_yx_array = []

	for vertex in vertices(mesh):
		point = vertex.point()
		P_xx_array.append(P_xx(point))
		P_yy_array.append(P_yy(point))
		P_xy_array.append(P_xy(point))
		P_yx_array.append(P_yx(point))

	# Move the mesh according to the displacement field u
	ALE.move(mesh, u)

	coordinates_after = mesh.coordinates()

	displacement = coordinates_after - coordinates_before

	return coordinates_before, displacement, P_xx_array, P_yy_array, P_xy_array, P_yx_array

