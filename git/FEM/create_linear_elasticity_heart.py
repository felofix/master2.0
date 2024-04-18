from dolfin import *
import fenics as fe
import cardiac_geometries

mu = 1.0
lambda_ = 0.5

def epsilon(u):
	# Engineering strain
	return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
	# Stress
	return lambda_ * div(u) * Identity(u.geometric_dimension()) + 2 * mu * epsilon(u)

def solve_linear_heart():
	filename = "/Users/Felix/desktop/New master folder/master2.0/git/heart_model/meshes/lv-mesh"
	geo = cardiac_geometries.geometry.Geometry.from_folder(filename)

	# Load the mesh.
	mesh = geo.mesh

	# Boundaries. 
	boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	boundaries.set_all(0)

	# Define markers for different regions
	endo_marker = 1
	epi_marker = 2
	base_marker = 3

	# Marking.
	for facet in facets(geo.mesh):
			if geo.ffun[facet] == geo.markers["ENDO"][0]:
				boundaries[facet] = endo_marker
			if geo.ffun[facet] == geo.markers["EPI"][0]:
				boundaries[facet] = epi_marker
			if geo.ffun[facet] == geo.markers["BASE"][0]:
				boundaries[facet] = base_marker
	coordinates_before = mesh.coordinates().copy()

	# Creating vector space for basis functions
	V = VectorFunctionSpace(mesh, "Lagrange", 1)
	u = TrialFunction(V)
	v = TestFunction(V)

	# Define boundary condition (Dirichlet)
	dirichlet_base = DirichletBC(V, Constant((0, 0, 0)), boundaries, 3)

	# Define measure for integrating over the boundary
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	a = inner(sigma(u), epsilon(v)) * dx
	
	# Assuming 'pressure' acts normal to the boundary, modify this as needed for your application
	#pressure = Constant(-1.0)
	pressure = Constant(0.3)
	n = FacetNormal(mesh)
	#f = fe.Constant((0, 0, -0.01))
	f = fe.Constant((0, 0, 0))
	L = fe.dot(f, v)*fe.dx + dot(pressure * n, v) * ds(1)  # Apply pressure on boundary with marker 1

	# Compute solution
	u = Function(V)  # Displacement field
	solve(a == L, u, dirichlet_base)

	S = TensorFunctionSpace(mesh, "Lagrange", 1)  # Tensor function space for sigma
	sigma_u = project(sigma(u), S)
	S_scalar = FunctionSpace(mesh, "Lagrange", 1)  # Scalar function space
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

	for vertex in vertices(mesh):
		point = vertex.point()
		s_xx_array.append(s_xx(point))
		s_yy_array.append(s_yy(point))
		s_zz_array.append(s_zz(point))
		s_xy_array.append(s_xy(point))
		s_xz_array.append(s_xz(point))
		s_yz_array.append(s_yz(point))


	# Move the mesh according to the displacement field u
	ALE.move(mesh, u)

	# Now mesh.coordinates() will return the new coordinates of the mesh vertices
	coordinates_after = mesh.coordinates()

	return coordinates_before, coordinates_after, coordinates_after - coordinates_before, s_xx_array, s_yy_array, s_zz_array, s_xy_array, s_xz_array, s_yz_array

cb, ca, g, s_xx_array, s_yy_array, s_zz_array, s_xy_array, s_xz_array, s_yz_array = solve_linear_heart()

with open('../data/heart_data/linear_elasticity_data.txt', 'w') as file:
	for i in range(len(s_xx_array)):
		file.write(f'{cb[:, 0][i]}, {cb[:, 1][i]}, {cb[:, 2][i]}, {g[:, 0][i]}, {g[:, 1][i]}, {g[:, 2][i]} ,{s_xx_array[i]}, {s_yy_array[i]}, {s_zz_array[i]}, {s_xy_array[i]}, {s_xz_array[i]}, {s_yz_array[i]}\n')