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
    parser.add_argument('--data', type=str, default='data/inflated_diffusion.csv')
    parser.add_argument('--output-diffusion', type=str)
    parser.add_argument('--output-heterogeneity', type=str)
    args = parser.parse_args()
    data = pd.read_csv(args.data)

    linear_bins=5
    Lx, Ly = 1, 1
    nbins=linear_bins**2

    density_variation = data.groupby(['interaction_radius', 'density_reg', 'D'])['density_variation'].mean()
    diffusion_mean = data.groupby(['interaction_radius', 'density_reg', 'D'])['meanD'].mean()
    diffusion_std = data.groupby(['interaction_radius', 'density_reg', 'D'])['stdD'].mean()
    D_array = data.D.unique()
    interaction_radius = data.interaction_radius.unique()
    density_reg = data.density_reg.unique()
    N_vals = [500] #data.N.unique()

    area = np.minimum(1,D_array*np.pi**2)

    ls = ['-', '-.', "--", ":"] #, "-", "--"]
    plt.figure()
    for N in N_vals:
        for i, (ir, dr) in enumerate(product(interaction_radius[1:], density_reg[:-2])):
            plt.plot(D_array*N/Lx/Ly, density_variation[ir, dr, :],
                    label=f'r={ir}, a={dr}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")

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
    for N in N_vals:
        for i, (ir, dr) in enumerate(product(interaction_radius[1:], density_reg[:-2])):
            plt.errorbar(D_array*N, diffusion_mean[ir, dr, :]/D_array,
                                    diffusion_mean[ir, dr, :]/D_array/np.sqrt(10),
                 label=f'r={ir}, a={dr}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")

    plt.legend()
    plt.plot(N*D_array, np.ones_like(D_array), c='k')
    plt.xlabel('true N*D')
    plt.xlabel('estimated N*D')
    # plt.yscale('log')
    plt.xscale('log')
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)
