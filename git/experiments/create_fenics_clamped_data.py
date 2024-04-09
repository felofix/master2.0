import sys
sys.path.append('../FEM/linear_elasticity_2D/')
import clamped_elasticity_fenics as cbf 

# Test plotting. 
cb, ca, u, s_xx, s_yy, s_xy = cbf.solve_clamped_beam_fenics(19, 19, 1)

with open('../data/clamped_beam/clamped_beam_fenics_info.txt', 'w') as file:
    for i in range(len(u)):
        file.write(f'{u[i][0]}, {u[i][1]}, {s_xx[i]}, {s_yy[i]}, {s_xy[i]}\n')

# Test plotting. 
cb, ca, u, s_xx, s_yy, s_xy = cbf.solve_clamped_beam_fenics(199, 199, 1)

with open('../data/clamped_beam/clamped_beam_fenics_beauty.txt', 'w') as file:
    for i in range(len(u)):
        file.write(f'{u[i][0]}, {u[i][1]}, {s_xx[i]}, {s_yy[i]}, {s_xy[i]}\n')