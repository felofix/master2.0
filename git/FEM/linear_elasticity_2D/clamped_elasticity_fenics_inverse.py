from fenics import *
from fenics_adjoint import *

# Create a rectangular mesh
length = 2.0
height = 1.0
nx = 40  # Number of elements along the length
ny = 20  # Number of elements along the height
mesh = RectangleMesh(Point(0, 0), Point(length, height), nx, ny)

# Define function space for vector-valued functions
V = VectorFunctionSpace(mesh, "Lagrange", degree=2)

# Lamé parameters
mu = Constant(1.0)
lmbda = Constant(0.5)

# Define boundary condition for clamped boundary (e.g., left side)
def left_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

bc = DirichletBC(V, Constant((0, 0)), left_boundary)

# Define strain and stress tensors
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda * div(u) * Identity(len(u)) + 2 * mu * epsilon(u)

# Define trial and test functions
u = Function(V, name="Displacement")
v = TestFunction(V)

# Define body forces (e.g., gravity)
f = Constant((0, -9.81))

# Define the variational problem
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx

# Solve the linear system
solve(a == L, u, bc)

# Compute total elastic energy
J = assemble(0.5 * inner(sigma(u), epsilon(u)) * dx)

# Define a control variable, e.g., the body force
control = Control(f)

# Compute the gradient of J with respect to the control
dJ_df = compute_gradient(J, control)

# Output results
print("Total elastic energy:", J)
print("Gradient of the functional with respect to the body force:", dJ_df)

# Save the displacement field to a file for visualization
File("displacement.pvd") << u
