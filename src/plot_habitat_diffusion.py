import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from heterogeneity import get_granularity
from itertools import product

def free_diffusion(D_array, N, linear_bins=5):
    n_iter = 10
    res = []
    for D in D_array:
        res.append([get_granularity(N, D, bins=linear_bins) for i in range(n_iter)])

    res = np.array(res)
    return res.mean(axis=1)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/habitat_diffusion.csv')
    parser.add_argument('--output-diffusion', type=str)
    parser.add_argument('--output-heterogeneity', type=str)
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    linear_bins=5
    Lx, Ly = 1, 1
    nbins=linear_bins**2

    density_variation = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['density_variation'].mean()
    diffusion_mean = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['meanD'].mean()
    diffusion_std = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['stdD'].mean()
    interaction_radius = data.interaction_radius.unique()
    period = data.period.unique()
    subsampling = data.subsampling.unique()
    density_reg = data.density_reg.unique()
    N_vals = data.N.unique()


    ls = ['-', '-.', "--"] #, ":"] #, "-", "--"]
    plt.figure()
    for N in N_vals:
        D_array = data.loc[data.N==N].D.unique()
        area = np.minimum(1,D_array*np.pi**2)
        for i, (ir, dr, T, p) in enumerate(product(interaction_radius, density_reg,
                                                   period, subsampling)):
            try:
                plt.plot(D_array*N/Lx/Ly, density_variation[ir, dr, :, N, T, p],
                        label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")
            except:
                pass

        free_diffusion_heterogeneity = free_diffusion(D_array*N/Lx/Ly, N, linear_bins=linear_bins)
        plt.plot(D_array*N/Lx/Ly, free_diffusion_heterogeneity, label='diffusion', c='k')
        plt.plot(D_array*N/Lx/Ly, np.ones_like(D_array)*np.sqrt(nbins/N), label='well mixed limit', c='k', ls='--')
    plt.xscale('log')
    plt.xlabel('diffusion constant')
    plt.ylabel('heterogeneity')
    plt.legend()
    if args.output_heterogeneity:
        plt.savefig(args.output_heterogeneity)


    plt.figure()
    for m, N in zip(['o', 'd'], N_vals):
        D_array = data.loc[data.N==N].D.unique()
        print(D_array[0])
        for i, (ir, dr, T, p) in enumerate(product(interaction_radius, density_reg,
                                                   period, subsampling)):
            try:
                plt.errorbar(D_array*N/Lx/Ly, diffusion_mean[ir, dr, :, N, T, p]/D_array,
                                    diffusion_std[ir, dr, :, N, T, p]/D_array/np.sqrt(10), marker=m,
                    label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")
            except:
                pass

        plt.plot(N*D_array/Lx/Ly, np.ones_like(D_array), c='k')
    plt.legend()
    plt.xlabel('true N*D')
    plt.xlabel('estimated N*D')
    plt.yscale('log')
    plt.xscale('log')
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)
