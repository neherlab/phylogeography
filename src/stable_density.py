import numpy as np
from density_regulation import make_node, evolve, clean_tree, subsample_tree
from heterogeneity import get_2d_hist
from estimate_diffusion_from_tree import estimate_diffusion, estimate_ancestral_positions, collect_zscore


def evolve_stable_density(D, interaction_radius, density_reg, N, subsampling=1.0,
                                Lx=1, Ly=1, linear_bins=5, n_iter=10, n_subsamples=1):
    from scipy.stats import scoreatpercentile
    # set up tree and initial population uniformly in space
    tree = make_node(Lx/2,Ly/2,-2, None)
    tree['children'] = [make_node(np.random.random()*Lx, np.random.random()*Ly, -1, tree)
                        for i in range(N)]
    terminal_nodes = tree['children']

    density_variation = []
    D_est = []
    zscores = []
    Tmrca = []
    for t in range((n_iter+10)*N):
        terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly, interaction_radius=interaction_radius,
                                density_reg=density_reg, D=D, target_density=N, total_population=N)
        if len(terminal_nodes)<10:
            print("population nearly extinct")
            continue
        if t%(N//5)==0 and t>10*N: # take samples after burnin every Tc//5
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
                    root_index = 1
                else:
                    Tmrca.append(t)
                    root_index = 0

                internal_node_times= sorted(z.loc[z.nonterminal, 't'])
                tbins = scoreatpercentile(internal_node_times, [0, 20, 40, 60, 80, 100])
                D_est.extend([D_res['Dx_total'], D_res['Dy_total']])
                # calculate the mean squared z-scores for the root node and each time bin
                zscores.extend([[z.iloc[root_index].zx**2]+[np.mean(z.loc[(z.t >= tbins[i]) & (z.t<tbins[i+1]), 'zx']**2) for i in range(len(tbins)-1)],
                                [z.iloc[root_index].zy**2]+[np.mean(z.loc[(z.t >= tbins[i]) & (z.t<tbins[i+1]), 'zy']**2) for i in range(len(tbins)-1)]])

    return {"density_variation": density_variation, "D_est": D_est, "zscores": zscores, "Tmrca":Tmrca}

if __name__=="__main__":
    import sys
    sys.setrecursionlimit(10000)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--interaction-radius', type=float, default=0.1)
    parser.add_argument('--density-reg', type=float, default=0.1)
    parser.add_argument('--subsampling', type=float, default=1.0)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    # Evolve a population with density regulation and estimate the diffusion constant, the TMRCA and the coverage of diffusion estimates

    N = args.N
    Lx, Ly = 1, 1
    res_density = {}
    res_density_mean = {}
    D_est = []
    D_array_dens = np.logspace(-3,0,21)*Lx*Ly*2/N
    n_iter = 50
    linear_bins=5
    interaction_radius, density_reg = args.interaction_radius, args.density_reg
    nsub = 1 if args.subsampling>0.9 else 5
    print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")
    for di, D in enumerate(D_array_dens):
        print(f"{di} out of {len(D_array_dens)}: D={D:1.3e}")
        res = evolve_stable_density(D, interaction_radius, density_reg, N, subsampling=args.subsampling,
                                          Lx=Lx, Ly=Ly, linear_bins=linear_bins, n_iter=n_iter, n_subsamples=nsub)
        tmpD = np.mean(res["D_est"], axis=0)
        tmpStdD = np.std(res["D_est"], axis=0)
        tmpZ =    f"[{' '.join(str(x) for x in np.ma.mean(np.ma.masked_invalid(res['zscores']), axis=0).filled(fill_value=np.nan))}]"
        tmpStdZ = f"[{' '.join(str(x) for x in np.ma.std(np.ma.masked_invalid(res['zscores']), axis=0).filled(fill_value=np.nan))}]"
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
        fname = f'data/stable_density_{N=}_ir={interaction_radius}_dr={density_reg}.csv'

    pd.DataFrame(D_est).to_csv(fname, index=False)

