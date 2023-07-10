import numpy as np
from itertools import product
from density_regulation import make_node, evolve
from heterogeneity import get_2d_hist, get_granularity
from estimate_diffusion_from_tree import estimate_diffusion
import matplotlib.pyplot as plt

if __name__=="__main__":
    N = 500
    Lx, Ly = 1, 1
    res_density = {}
    res_density_mean = {}
    D_est = {}
    D_array_dens = np.logspace(-2,0,6)*Lx*Ly*2/N
    n_iter = 10
    linear_bins=5
    for interaction_radius, density_reg in product([0.02, 0.05, 0.1, 0.5, 2], [0.05, 0.1, 0.2]):
        print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")
        label = (interaction_radius, density_reg)
        tmp_density = []
        tmp_D_est = []
        for D in D_array_dens:
            tree = make_node(Lx/2,Ly/2,-2, None)
            tree['children'] = [make_node(np.random.random()*Lx, np.random.random()*Ly, -1, tree)
                                for i in range(N)]
            terminal_nodes = tree['children']
            tmp_density_local = []
            tmp_D_est_local = []
            for t in range((n_iter+2)*N):
                terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly, interaction_radius=interaction_radius,
                                        density_reg=density_reg, D=D, target_density=N)
                if t%(N)==0 and t>2*N:
                    H, bx, by = get_2d_hist(terminal_nodes, Lx, Ly, linear_bins)
                    tmp_density_local.append(np.std(H)/N*np.prod(H.shape))
                    D_res = estimate_diffusion(tree)
                    tmp_D_est_local.extend([D_res['Dx_total'], D_res['Dy_total']])
            tmp_density.append(tmp_density_local)
            tmp_D_est.append(np.mean(tmp_D_est_local, axis=0))
        res_density[label] = np.array(tmp_density)
        res_density_mean[label] = res_density[label].mean(axis=1)
        D_est[label] = np.array(tmp_D_est)


    nbins=linear_bins**2
    res = []
    D_array = np.logspace(-2,2,101)
    for D in D_array:
        res.append([get_granularity(N, D, bins=linear_bins) for i in range(n_iter)])

    res = np.array(res)
    res_mean = res.mean(axis=1)

    area = np.minimum(1,D_array*np.pi**2)

    ls = ['-', '-.', "--"]
    for i, dr in enumerate(res_density_mean):
        plt.plot(D_array_dens*N/Lx/Ly, res_density_mean[dr],
                 label=f'r={dr[0]}, a={dr[1]}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")
    plt.plot(D_array, res_mean, label='diffusion')
    plt.plot(D_array, np.ones_like(D_array)*np.sqrt(nbins/N), label='well mixed limit')
    plt.xscale('log')
    plt.xlabel('diffusion constant')
    plt.ylabel('heterogeneity')
    plt.legend()

    plt.figure()
    for i, dr in enumerate(res_density_mean):
        plt.plot(D_array_dens*N, D_est[dr]/D_array_dens,
                 label=f'r={dr[0]}, a={dr[1]}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")

    plt.legend()
    plt.plot(N*D_array_dens, np.ones_like(D_array_dens), c='k')
    plt.xlabel('true N*D')
    plt.xlabel('estimated N*D')
    plt.yscale('log')
    plt.xscale('log')
