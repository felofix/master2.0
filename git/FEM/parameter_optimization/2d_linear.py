from dolfin import *
from dolfin_adjoint import *

# Silence log output
set_log_level(LogLevel.ERROR)

nx, ny = 20, 20
mesh = UnitSquareMesh(nx, ny)
dx_ = Measure("dx", domain=mesh)  # Explicit measure on this mesh

# Define the displacement function space directly
V = VectorFunctionSpace(mesh, "CG", 1)  # Displacement space

# Define control parameters as Constants
mu = Constant(0.8)  # Initial guess for shear modulus
lambda_ = Constant(0.3)  # Initial guess for Lame's first parameter

def epsilon(u):
    return 0.5 * (grad(u) + grad(u).T)

def sigma(u, mu, lambda_):
    return lambda_ * div(u) * Identity(len(u)) + 2.0 * mu * epsilon(u)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)

LeftBoundary().mark(boundary_markers, 1)
RightBoundary().mark(boundary_markers, 2)

# Define Dirichlet boundary conditions
left_bc = DirichletBC(V, Constant((0.0, 0.0)), boundary_markers, 1)
right_bc = DirichletBC(V, Constant((0.0, 0.0)), boundary_markers, 2)
bcs = [left_bc, right_bc]

# Body force
f = Constant((0.0, -1.0))  # Body force acting downward

# Define the forward problem
u = Function(V, name="Displacement")
du = TrialFunction(V)  # Incremental displacement
v = TestFunction(V)    # Test function

# Define the variational form
F = inner(sigma(u, mu, lambda_), epsilon(v)) * dx - dot(f, v) * dx

# Compute Jacobian form
J = derivative(F, u, du)

# Create problem and solver
problem = NonlinearVariationalProblem(F, u, bcs, J)
solver = NonlinearVariationalSolver(problem)

# Set solver parameters
solver_parameters = {"newton_solver": {"linear_solver": "mumps",
                                     "relative_tolerance": 1e-6,
                                     "maximum_iterations": 100}}
solver.parameters.update(solver_parameters)

# Solve the forward problem
solver.solve()

# Define the objective functional
alpha = Constant(10.0)  # Regularization parameter
J_func = Function(0.5 * inner(u, u) * dx_ + alpha / 2 * (mu**2 + lambda_**2) * dx_)

# Define the reduced functional
control_mu = Control(mu)
control_lambda = Control(lambda_)
rf = ReducedFunctional(J_func, [control_mu, control_lambda])

# Optimize
opt_params = minimize(rf, method="L-BFGS-B", 
                     options={"disp": True, "gtol": 1e-10, "ftol": 1e-10})

# Update the parameters with optimal values
mu.assign(opt_params[0])
lambda_.assign(opt_params[1])

# Solve the forward problem again with optimal parameters
solver.solve()

# Print final optimized parameters
print(f"Optimized mu = {float(mu)}")
print(f"Optimized lambda = {float(lambda_)}")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot displacement magnitude
plt.subplot(1, 2, 1)
u_magnitude = sqrt(dot(u, u))
p1 = plot(u_magnitude)
plt.colorbar(p1)
plt.title("Displacement magnitude")

# Plot displacement vector
plt.subplot(1, 2, 2)
p2 = plot(u)
plt.title("Displacement vector field")

plt.tight_layout()
plt.savefig("displacement_results.png")
plt.close()