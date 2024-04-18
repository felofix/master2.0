import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import dolfin
import ufl

# Constants
mu = 1.0
kappa = 0.5

def neo_hookean(F: ufl.Coefficient) -> ufl.Coefficient:
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
	I1 = dolfin.tr(C)
	return 0.5 * mu * (I1 - 3)

def compressibility(F: ufl.Coefficient) -> ufl.Coefficient:
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
	J = dolfin.det(F)
	return kappa/2*(dolfin.ln(J))**2 - mu*dolfin.ln(J)

	# Boundaries. 
class Left(dolfin.SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and dolfin.near(x[0], 0)

class Right(dolfin.SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and dolfin.near(x[0], 1)

class Bottom(dolfin.SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and dolfin.near(x[2], 0)

class Top(dolfin.SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and dolfin.near(x[2], 1)

class Front(dolfin.SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and dolfin.near(x[1], 0)

class Back(dolfin.SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and dolfin.near(x[1], 1)

def solve_neo_hookian_3d_fenics(nx, ny, nz):

	# Boundaries and domain.
	left = Left()
	top = Top()
	right = Right()
	bottom = Bottom()

	# Mesh and Vector Function Space
	mesh = dolfin.UnitCubeMesh(nx, ny, nz)

	boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	boundaries.set_all(0)

	left.mark(boundaries, 1)
	top.mark(boundaries, 2)
	right.mark(boundaries, 3)
	bottom.mark(boundaries, 4)

	# Get the coordinates of the vertices before deformation.
	coordinates_before = mesh.coordinates().copy()

	# Creating vector space for basis functions.
	V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 2)
	u = dolfin.Function(V)
	v = dolfin.TestFunction(V)

	# Compute the deformation gradient
	F = dolfin.grad(u) + dolfin.Identity(3)

	J = dolfin.det(F)

	elastic_energy = neo_hookean(F) + compressibility(F)

	# Dirichlet bouondaries. 
	dirichlet_left  = dolfin.DirichletBC(V, dolfin.Constant((0, 0, 0)), boundaries, 1)
	dirichlet_right  = dolfin.DirichletBC(V, dolfin.Constant((0, 0, 0.0)), boundaries, 3)
	bcs = [dirichlet_left]

	# Neumann.
	traction = dolfin.Constant(0.3)
	N = dolfin.FacetNormal(mesh)
	n = traction * J*ufl.inv(F).T * N

	ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundaries)
	# Define new measures associated with the interior domains and
	# exterior boundaries
	quad_degree = 4
	internal_virtual_work = dolfin.derivative(
		elastic_energy * dolfin.dx(metadata={"quadrature_degree": quad_degree}), u, v
	)

	external_virtual_work = dolfin.inner(v, n) * ds(2)

	total_virtual_work = internal_virtual_work + external_virtual_work

	dolfin.solve(total_virtual_work == 0, u, bcs=bcs)

	# Compute P after solving the problem
	grader = dolfin.grad(u)
	P = mu * F + (kappa*dolfin.ln(J) - mu)*ufl.inv(ufl.transpose(F)) # compressability here. 
	S = dolfin.TensorFunctionSpace(mesh, "Lagrange", 1)
	P = dolfin.project(P, S)
	S_scalar = dolfin.FunctionSpace(mesh, "Lagrange", 1)  # Scalar function space

	P_xx = dolfin.project(P[0, 0], S_scalar)
	P_yy = dolfin.project(P[1, 1], S_scalar)
	P_zz = dolfin.project(P[2, 2], S_scalar)
	P_xy = dolfin.project(P[0, 1], S_scalar)
	P_yx = dolfin.project(P[1, 0], S_scalar)
	P_xz = dolfin.project(P[0, 2], S_scalar)
	P_zx = dolfin.project(P[2, 0], S_scalar)
	P_yz = dolfin.project(P[1, 2], S_scalar)
	P_zy = dolfin.project(P[2, 1], S_scalar)

	P_xx_array = []
	P_yy_array = []
	P_zz_array = []
	P_xy_array = []
	P_yx_array = []
	P_xz_array = []
	P_zx_array = []
	P_yz_array = []
	P_zy_array = []

	for vertex in dolfin.vertices(mesh):
		point = vertex.point()
		P_xx_array.append(P_xx(point))
		P_yy_array.append(P_yy(point))
		P_zz_array.append(P_zz(point))
		P_xy_array.append(P_xy(point))
		P_yx_array.append(P_yx(point))
		P_xz_array.append(P_xz(point))
		P_zx_array.append(P_zx(point))
		P_yz_array.append(P_yz(point))
		P_zy_array.append(P_zy(point))

	# Move the mesh according to the displacement field u
	dolfin.ALE.move(mesh, u)

	# Now mesh.coordinates() will return the new coordinates of the mesh vertices
	coordinates_after = mesh.coordinates()

	return coordinates_before, coordinates_after, coordinates_after - coordinates_before, P_xx_array, P_yy_array, P_zz_array, P_xy_array, P_yx_array, P_xz_array, P_zx_array, P_yz_array, P_zy_array