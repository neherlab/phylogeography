import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from heterogeneity import get_granularity
from itertools import product


def parse_data(data, groupby=None):
    density_variation = data.groupby(groupby + ['D'])['density_variation'].mean()
    nobs = data.groupby(groupby)['n'].mean()/25
    diffusion_mean = data.groupby(groupby + ['D'])['meanD'].mean()
    diffusion_std = data.groupby(groupby + ['D'])['stdD'].mean()
    tmrca_mean = data.groupby(groupby + ['D'])['meanTmrca'].mean()
    tmrca_std = data.groupby(groupby + ['D'])['stdTmrca'].mean()

    z_mean = {}
    for g, d in data.groupby(groupby + ['D'])['meanZsq']:
        z_mean[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    z_std = {}
    for g, d in data.groupby(groupby + ['D'])['stdZsq']:
        z_std[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    return {"density_variation": density_variation, "nobs":nobs,
            "diffusion_mean":diffusion_mean, "diffusion_std":diffusion_std,
            "tmrca_mean":tmrca_mean, "tmrca_std":tmrca_std,
            "z_mean": pd.DataFrame(z_mean).T, "z_std":pd.DataFrame(z_std).T,
            }


def free_diffusion(D_array, N, linear_bins=5):
    n_iter = 10
    res = []
    for D in D_array:
        res.append([get_granularity(N, D, bins=linear_bins)[0] for i in range(n_iter)])

    res = np.array(res)
    return res.mean(axis=1)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/inflated_diffusion.csv')
    parser.add_argument('--output-diffusion', type=str)
    parser.add_argument('--output-heterogeneity', type=str)
    parser.add_argument('--output-zscore', type=str)
    parser.add_argument('--output-tmrca', type=str)
    args = parser.parse_args()
    data = pd.read_csv(args.data)

    linear_bins=5
    Lx, Ly = 1, 1
    nbins=linear_bins**2
    res = parse_data(data, groupby=['interaction_radius', 'density_reg', 'N'])

    interaction_radius = data.interaction_radius.unique()
    density_reg = data.density_reg.unique()
    N_vals = data.N.unique()

    ir_to_plot = interaction_radius[:]
    density_reg_to_plot = density_reg[:-2]

    ls = ['-', '-.', "--", ":"][:len(density_reg_to_plot)]
    plt.figure()
    for N in N_vals:
        for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
            D_array = np.array(res["density_variation"][ir, dr, N, :].index)
            label = f'r={ir}, a={dr}' if N==N_vals[0] else ''
            plt.plot(D_array*N/Lx/Ly, res["density_variation"][ir, dr, N, :],
                    label=label, ls=ls[i%len(ls)], c=f"C{i//len(ls)}")

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
        for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
            D_array = np.array(res["diffusion_mean"][ir, dr, N, :].index)
            label = f'r={ir}, a={dr}' if N==N_vals[0] else ''
            plt.errorbar(D_array*N, res["diffusion_mean"][ir, dr, N, :]/D_array,
                                    res["diffusion_std"][ir, dr, N, :]/D_array/np.sqrt(res["nobs"][ir, dr, N]),
                 label=label, ls=ls[i%len(ls)], c=f"C{i//len(ls)}")

    plt.legend()
    plt.plot(N*D_array, np.ones_like(D_array), c='k')
    plt.xlabel('true N*D')
    plt.ylabel('fold-error in D estimate')
    # plt.yscale('log')
    plt.xscale('log')
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)

    plt.figure()
    for N in N_vals:
        for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
            D_array = np.array(res["tmrca_mean"][ir, dr, N, :].index)
            label = f'r={ir}, a={dr}' if N==N_vals[0] else ''
            plt.errorbar(D_array*N, np.array([res["z_mean"].loc[(ir, dr, N, d)] for d in D_array]).mean(axis=1),
                                    np.array([res["z_std"].loc[(ir, dr, N, d)]/np.sqrt(res["nobs"][ir, dr, N]) for d in D_array]).mean(axis=1),
                 label=label, ls=ls[i%len(ls)], c=f"C{i//len(ls)}")

    plt.legend()
    plt.plot(N*D_array, np.ones_like(D_array), c='k')
    plt.xlabel('true N*D')
    plt.ylabel('Coverage')
    # plt.yscale('log')
    plt.xscale('log')
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)

    plt.figure()
    for N in N_vals:
        for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
            D_array = np.array(res["tmrca_mean"][ir, dr, N, :].index)
            label = f'r={ir}, a={dr}' if N==N_vals[0] else ''
            plt.errorbar(D_array*N, res["tmrca_mean"][ir, dr, N, :]/N/2,
                                    res["tmrca_std"][ir, dr, N, :]/N/2/np.sqrt(res["nobs"][ir, dr, N]),
                 label=label, ls=ls[i%len(ls)], c=f"C{i//len(ls)}")

    plt.legend()
    plt.plot(N*D_array, np.ones_like(D_array), c='k')
    plt.xlabel('true N*D')
    plt.ylabel('T_mrca/2N')
    # plt.yscale('log')
    plt.xscale('log')
    if args.output_tmrca:
        plt.savefig(args.output_tmrca)
