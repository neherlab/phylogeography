import numpy as np
from itertools import product
from density_regulation import make_node, evolve
from heterogeneity import get_2d_hist, get_granularity
from estimate_diffusion_from_tree import estimate_diffusion
import matplotlib.pyplot as plt

def estimate_inflated_diffusion(D, interaction_radius, density_reg, N, Lx=1, Ly=1, linear_bins=5, n_iter=10):
    # set up tree and initial population uniformly in space
    tree = make_node(Lx/2,Ly/2,-2, None)
    tree['children'] = [make_node(np.random.random()*Lx, np.random.random()*Ly, -1, tree)
                        for i in range(N)]
    terminal_nodes = tree['children']

    density_variation = []
    D_est = []
    for t in range((n_iter+2)*N):
        terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly, interaction_radius=interaction_radius,
                                density_reg=density_reg, D=D, target_density=N)
        if t%(N//5)==0 and t>2*N: # take samples after burnin every 5 Tc
            H, bx, by = get_2d_hist(terminal_nodes, Lx, Ly, linear_bins)
            density_variation.append(np.std(H)/N*np.prod(H.shape))
            D_res = estimate_diffusion(tree)
            D_est.extend([D_res['Dx_total'], D_res['Dy_total']])

    return {"density_variation": density_variation, "D_est": D_est}

if __name__=="__main__":
    N = 500
    Lx, Ly = 1, 1
    res_density = {}
    res_density_mean = {}
    D_est = []
    D_array_dens = np.logspace(-2,0,6)*Lx*Ly*2/N
    n_iter = 10
    linear_bins=5
    iR, dR = [0.02, 0.05, 0.1, 0.5, 2], [0.05, 0.1, 0.2]
    iR, dR = [0.02], [0.05]
    for interaction_radius, density_reg in product(iR, dR):
        print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")
        for D in D_array_dens:
            res = estimate_inflated_diffusion(D, interaction_radius, density_reg, N, Lx=Lx, Ly=Ly, linear_bins=linear_bins, n_iter=n_iter)
            Dx, Dy = np.mean(res["D_est"], axis=0)
            stdDx, stdDy = np.std(res["D_est"], axis=0)
            D_est.append({"interaction_radius":interaction_radius, "density_reg": density_reg,
                              "D":D, "meanD_x": Dx, "meanD_y": Dx, "stdDx": stdDx, "stdDy": stdDy, "density_variation": res['density_variation'].mean()})

    import pandas as pd
    pd.DataFrame(D_est).to_csv('data/inflated_diffusion_{N=}.csv')

