import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import dolfin
import ufl
import cardiac_geometries

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

def active_stress_energy(F: ufl.Coefficient, f0: dolfin.Function, Ta: dolfin.Constant) -> ufl.Coefficient:
    """Active stress energy
    Parameters
    ----------
    F : ufl.Coefficient
        Deformation gradient
    f0 : dolfin.Function
        Fiber direction
    Ta : dolfin.Constant
        Active tension
    Returns
    -------
    ufl.Coefficient
        Active stress energy
    """

    I4f = dolfin.inner(F * f0, F * f0)
    return 0.5 * Ta * (I4f - 1)

def solve_neo_hookian_heart(active_stress = False):
	filename = "/Users/Felix/desktop/New master folder/master2.0/git/heart_model/meshes/lv-mesh"
	geo = cardiac_geometries.geometry.Geometry.from_folder(filename)

	# Load the mesh.
	mesh = geo.mesh

	# Boundaries. 
	boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
	boundaries.set_all(0)

	# Define markers for different regions
	endo_marker = 1
	epi_marker = 2
	base_marker = 3

	# Active tension
	Ta = dolfin.Constant(1.0)
	# Set fiber direction to be constant in the x-direction
	if active_stress == True:
		f0 = geo.f0
	else:
		f0 = 0

	# Marking.
	for facet in dolfin.facets(geo.mesh):
			if geo.ffun[facet] == geo.markers["ENDO"][0]:
				boundaries[facet] = endo_marker
			if geo.ffun[facet] == geo.markers["EPI"][0]:
				boundaries[facet] = epi_marker
			if geo.ffun[facet] == geo.markers["BASE"][0]:
				boundaries[facet] = base_marker

	coordinates_before = mesh.coordinates().copy()

	# Creating vector space for basis functions.
	V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 2)
	u = dolfin.Function(V)
	v = dolfin.TestFunction(V)

	# Compute the deformation gradient
	F = dolfin.grad(u) + dolfin.Identity(3)

	J = dolfin.det(F)

	elastic_energy = neo_hookean(F) + compressibility(F) + active_stress_energy(F, f0, Ta)

	# Dirichlet bouondaries. 
	dirichlet_base = dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), boundaries, 3)
	ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundaries)

	# Neumann.
	traction = dolfin.Constant(0.3)
	N = dolfin.FacetNormal(mesh)
	n = traction * J*ufl.inv(F).T * N

	# Define new measures associated with the interior domains and
	quad_degree = 4
	internal_virtual_work = dolfin.derivative(
	    elastic_energy * dolfin.dx(metadata={"quadrature_degree": quad_degree}), u, v
	)

	external_virtual_work = dolfin.inner(v, n) * ds(1)

	total_virtual_work = internal_virtual_work + external_virtual_work

	dolfin.solve(total_virtual_work == 0, u, dirichlet_base)

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

	f0_array = []

	for vertex in dolfin.vertices(mesh):
		point = vertex.point()
		if active_stress == True:
			f0_array.append(f0(point))

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

	return coordinates_before, coordinates_after, coordinates_after - coordinates_before, P_xx_array, P_yy_array, P_zz_array, P_xy_array, P_yx_array, P_xz_array, P_zx_array, P_yz_array, P_zy_array, np.array(f0_array)


cb, ca, g, P_xx_array, P_yy_array, P_zz_array, P_xy_array, P_yx_array, P_xz_array, P_zx_array, P_yz_array, P_zy_array, f0 = solve_neo_hookian_heart()

with open('../data/heart_data/hyper_elasticity_data.txt', 'w') as file:
    for i in range(len(P_xx_array)):
        file.write(f'{cb[:, 0][i]}, {cb[:, 1][i]}, {cb[:, 2][i]}, {g[:, 0][i]}, {g[:, 1][i]}, {g[:, 2][i]} ,{P_xx_array[i]}, {P_yy_array[i]}, {P_zz_array[i]}, {P_xy_array[i]}, {P_yx_array[i]}, {P_xz_array[i]}, {P_zx_array[i]}, {P_yz_array[i]},  {P_zy_array[i]}\n')

cb, ca, g, P_xx_array, P_yy_array, P_zz_array, P_xy_array, P_yx_array, P_xz_array, P_zx_array, P_yz_array, P_zy_array, f0 = solve_neo_hookian_heart(active_stress = True)


with open('../data/heart_data/hyper_elasticity_data_active.txt', 'w') as file:
    for i in range(len(P_xx_array)):
        file.write(f'{cb[:, 0][i]}, {cb[:, 1][i]}, {cb[:, 2][i]}, {g[:, 0][i]}, {g[:, 1][i]}, {g[:, 2][i]} ,{P_xx_array[i]}, {P_yy_array[i]}, {P_zz_array[i]}, {P_xy_array[i]}, {P_yx_array[i]}, {P_xz_array[i]}, {P_zx_array[i]}, {P_yz_array[i]},  {P_zy_array[i]}, {f0[:, 0][i]}, {f0[:, 1][i]} ,{f0[:, 2][i]}\n')