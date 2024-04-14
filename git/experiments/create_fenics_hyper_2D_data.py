import sys
sys.path.append('../FEM/hyper_elasticity_2D/')
import hyper_elasticity_fenics_2D as hef 

# Test plotting. 
cb, g, pxx, pyy, pxy, pyx = hef.solve_neo_hookian(19, 19)

with open('../data/hyper_beam/hyper_beam_fenics_info.txt', 'w') as file:
    for i in range(len(pyx)):
        file.write(f'{cb[:, 0][i]}, {cb[:, 1][i]}, {g[:, 0][i]}, {g[:, 1][i]},\
                     {pxx[i]}, {pxy[i]}, {pyx[i]}, {pyy[i]}\n')

# Test plotting. 
cb, g, pxx, pyy, pxy, pyx = hef.solve_neo_hookian(99, 99)

with open('../data/hyper_beam/hyper_beam_fenics_beauty.txt', 'w') as file:
    for i in range(len(pyx)):
        file.write(f'{cb[:, 0][i]}, {cb[:, 1][i]}, {g[:, 0][i]}, {g[:, 1][i]},\
                     {pxx[i]}, {pxy[i]}, {pyx[i]}, {pyy[i]}\n')