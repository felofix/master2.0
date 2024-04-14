import sys
sys.path.append('../FEM/linear_elasticity_3D/')
import clamped_3d as scbf 

cb, ca, g, s_xx_array, s_yy_array, s_zz_array, s_xy_array, s_xz_array, s_yz_array, u_xx_array, u_yy_array, u_zz_array, u_xy_array, u_yx_array, u_xz_array, u_zx_array, u_yz_array, u_zy_array = scbf.solve_clamped_beam_fenics(10, 10, 10)

with open('../data/clamped_beam_3D/linear_data_fenics.txt', 'w') as file:
    for i in range(len(s_xx_array)):
        file.write(f'{cb[:, 0][i]}, {cb[:, 1][i]}, {cb[:, 2][i]}, {g[:, 0][i]}, {g[:, 1][i]}, {g[:, 2][i]}, \
                     {s_xx_array[i]}, {s_yy_array[i]}, {s_zz_array[i]}, {s_xy_array[i]}, {s_xz_array[i]}, {s_yz_array[i]}, \
                     {u_xx_array[i]}, {u_yy_array[i]}, {u_zz_array[i]}, {u_xy_array[i]}, {u_yx_array[i]}, {u_xz_array[i]}, \
                     {u_zx_array[i]}, {u_yz_array[i]}, {u_zy_array[i]}\n')