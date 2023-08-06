import numpy as np
from itertools import product
from density_regulation import make_node, evolve, calc_density
from heterogeneity import get_2d_hist
from estimate_diffusion_from_tree import estimate_diffusion
import matplotlib.pyplot as plt

if __name__=="__main__":
    N = 200
    Lx, Ly = 4, 1
    res_density = {}
    res_density_mean = {}
    D_est = {}
    D_array_dens = np.logspace(-2,0,6)*Ly**2/N
    n_iter = 1
    linear_bins=5

    for interaction_radius, density_reg in product([0.1], [0.2]):
        print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")
        label = (interaction_radius, density_reg)
        tmp_density = []
        tmp_D_est = []
        for D in [0.0001]:
            tree = make_node(Lx/2,Ly/2,-2, None)
            # confine population to a 10% stripe in x
            tree['children'] = [make_node((np.random.random()*interaction_radius-0.5*interaction_radius) + Lx/2,
                                          np.random.random()*Ly, -1, tree) for i in range(N)]
            terminal_nodes = tree['children']
            tmp_density_local = []
            tmp_D_est_local = []
            for t in range(501):
                terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly,
                                        interaction_radius=interaction_radius, density_reg=density_reg,
                                        D=D, target_density=N/interaction_radius, global_pop_reg=False)
                if t%100==0:
                    x_pos = np.array([c['x'] for c in terminal_nodes])%Lx
                    y_pos = np.array([c['y'] for c in terminal_nodes])%Ly
                    print(f"{t=}, pop_size={len(terminal_nodes)}, x_range={np.std(x_pos):1.3f}")
                    plt.figure()
                    h=plt.hexbin(x_pos, y_pos, gridsize=50, extent=(0,Lx,0,Ly))
                if t%(N)==0 and t>N:
                    H, bx, by = get_2d_hist(terminal_nodes, Lx, Ly, linear_bins)
                    tmp_density_local.append(np.std(H)/N*np.prod(H.shape))
                    D_res = estimate_diffusion(tree)
                    tmp_D_est_local.append([D_res['Dx_total'], D_res['Dy_total']])
            tmp_density.append(tmp_density_local)
            tmp_D_est.append(tmp_D_est_local)
        res_density[label] = np.array(tmp_density)
        res_density_mean[label] = res_density[label].mean(axis=1)
        D_est[label] = np.array(tmp_D_est)
