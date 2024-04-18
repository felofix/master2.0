import sys
sys.path.append('../FEM/hyper_elasticity_3D/')
import neo_hookian_3d as nh3d 

cb, ca, g, P_xx_array, P_yy_array, P_zz_array, P_xy_array, P_yx_array, P_xz_array, P_zx_array, P_yz_array, P_zy_array = nh3d.solve_neo_hookian_3d_fenics(9, 9, 9)

with open('../data/hyper_beam_3d/neo_hookian_3d_fenics.txt', 'w') as file:
    for i in range(len(P_xx_array)):
        file.write(f'{cb[:, 0][i]}, {cb[:, 1][i]}, {cb[:, 2][i]}, {g[:, 0][i]}, {g[:, 1][i]}, {g[:, 2][i]} ,{P_xx_array[i]}, {P_yy_array[i]}, {P_zz_array[i]}, {P_xy_array[i]}, {P_yx_array[i]}, {P_xz_array[i]}, {P_zx_array[i]}, {P_yz_array[i]},  {P_zy_array[i]}\n')