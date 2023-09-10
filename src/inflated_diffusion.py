import numpy as np
from density_regulation import make_node, evolve, clean_tree, subsample_tree
from heterogeneity import get_2d_hist
from estimate_diffusion_from_tree import estimate_diffusion, estimate_ancestral_positions, collect_zscore


def estimate_inflated_diffusion(D, interaction_radius, density_reg, N, subsampling=1.0,
                                Lx=1, Ly=1, linear_bins=5, n_iter=10, n_subsamples=1):
    # set up tree and initial population uniformly in space
    tree = make_node(Lx/2,Ly/2,-2, None)
    tree['children'] = [make_node(np.random.random()*Lx, np.random.random()*Ly, -1, tree)
                        for i in range(N)]
    terminal_nodes = tree['children']

    density_variation = []
    D_est = []
    zscores = []
    Tmrca = []
    for t in range((n_iter+5)*N):
        terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly, interaction_radius=interaction_radius,
                                density_reg=density_reg, D=D, target_density=N)
        if len(terminal_nodes)<10:
            print("population nearly extinct")
            continue
        if t%(N//5)==0 and t>5*N: # take samples after burnin every Tc//5
            clean_tree(tree)
            H, bx, by = get_2d_hist(terminal_nodes, Lx, Ly, linear_bins)
            density_variation.append(np.std(H)/N*np.prod(H.shape))
            for sample in range(n_subsamples):
                subsample_tree(terminal_nodes, tree, p=subsampling, subtree_attr='clades')
                D_res = estimate_diffusion(tree)
                estimate_ancestral_positions(tree, D)
                z = collect_zscore(tree)
                if len(tree['clades'])==1:
                    Tmrca.append(t-tree['clades'][0]['time'])
                else:
                    Tmrca.append(t)
                D_est.extend([D_res['Dx_total'], D_res['Dy_total']])
                zscores.extend([np.mean(z.loc[z.nonterminal, 'zx']**2),
                                np.mean(z.loc[z.nonterminal, 'zy']**2)])

    return {"density_variation": density_variation, "D_est": D_est, "zscores": zscores, "Tmrca":Tmrca}

if __name__=="__main__":
    import sys
    sys.setrecursionlimit(10000)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--D', type=float, default=0.1)
    parser.add_argument('--interaction-radius', type=float, default=0.1)
    parser.add_argument('--density-reg', type=float, default=0.1)
    parser.add_argument('--subsampling', type=float, default=1.0)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    N = args.N
    Lx, Ly = 1, 1
    res_density = {}
    res_density_mean = {}
    D_est = []
    D_array_dens = np.logspace(-1,0,1)*Lx*Ly*2/N
    n_iter = 1
    linear_bins=5
    interaction_radius, density_reg = args.interaction_radius, args.density_reg
    nsub = 1 if args.subsampling>0.9 else 5
    print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")
    for di, D in enumerate(D_array_dens):
        print(f"{di} out of {len(D_array_dens)}: D={D:1.3e}")
        res = estimate_inflated_diffusion(D, interaction_radius, density_reg, N, subsampling=args.subsampling,
                                          Lx=Lx, Ly=Ly, linear_bins=linear_bins, n_iter=n_iter, n_subsamples=nsub)
        tmpD = np.mean(res["D_est"], axis=0)
        tmpStdD = np.std(res["D_est"], axis=0)
        tmpZ = np.mean(res["zscores"], axis=0)
        tmpStdZ = np.std(res["zscores"], axis=0)
        nobs = len(res["D_est"])
        D_est.append({"interaction_radius":interaction_radius, "density_reg": density_reg,
                      "N": N, "n": len(res["D_est"]), "subsampling": args.subsampling,
                      "D":D, "meanD": tmpD, "stdD": tmpStdD,
                      "meanZsq": tmpZ, "stdZsq": tmpStdZ, "observations": nobs,
                      "density_variation": np.mean(res['density_variation']),
                      "meanTmrca":np.mean(res["Tmrca"]), "stdTmrca":np.std(res["Tmrca"])})

    import pandas as pd
    if args.output:
        fname = args.output
    else:
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        fname = f'data/inflated_diffusion_{N=}_ir={interaction_radius}_dr={density_reg}.csv'

    pd.DataFrame(D_est).to_csv(fname, index=False)

