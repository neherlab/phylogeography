import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from heterogeneity import get_granularity
from itertools import product
from parse_and_plot import parse_data

def free_diffusion(D_array, N, linear_bins=5):
    n_iter = 100
    res = []
    for D in D_array:
        res.append([get_granularity(N, D, bins=linear_bins)[0] for i in range(n_iter)])

    res = np.array(res)
    return res.mean(axis=1)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/stable_density.csv')
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

    # select the interaction radius and density regulation strength to plot
    ir_to_plot = interaction_radius[1:]
    density_reg_to_plot = density_reg[1:2]

    # Plot the heterogeneity
    ls = ['-', '-.', "--", ":"] #[:len(density_reg_to_plot)]
    plt.figure()
    for N in N_vals[1:2]:
        for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
            D_array = np.array(res["density_variation"][ir, dr, N, :].index)
            label = f'r={ir}' if N==N_vals[1] else ''
            plt.plot(D_array*N/Lx/Ly, res["density_variation"][ir, dr, N, :]/np.sqrt(nbins/N),
                    label=label, ls=ls[i%len(ls)], c=f"C{i%10}")

        free_diffusion_heterogeneity = free_diffusion(D_array*N/Lx/Ly, N, linear_bins=linear_bins)
    plt.plot(D_array*N/Lx/Ly, free_diffusion_heterogeneity/np.sqrt(nbins/N), label='free diffusion', c='k')
    plt.plot(D_array*N/Lx/Ly, np.ones_like(D_array), c='k', ls='--')
    plt.xscale('log')
    plt.xlabel('diffusion constant $[L^2/N]$')
    plt.ylabel('heterogeneity (relative to well-mixed case)')
    plt.legend()
    if args.output_heterogeneity:
        plt.savefig(args.output_heterogeneity)

    ## Plot the diffusion estiamtes
    plt.figure()
    for ni,N in enumerate(N_vals):
        for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
            D_array = np.array(res["diffusion_mean"][ir, dr, N, :].index)
            label = f'r={ir}, a={dr}' if N==N_vals[0] else ''
            plt.errorbar(D_array*N, res["diffusion_mean"][ir, dr, N, :]/D_array,
                                    res["diffusion_std"][ir, dr, N, :]/D_array/np.sqrt(res["nobs"][ir, dr, N]),
                 label=label, ls=ls[ni%len(ls)], c=f"C{i%10}")

    plt.legend()
    plt.plot(N*D_array, np.ones_like(D_array), c='k')
    plt.xlabel('true N*D')
    plt.ylabel('fold-error in D estimate')
    # plt.yscale('log')
    plt.xscale('log')
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)

    # plot the z-scores, i.e. the degree to which the estimates cover the true value
    plt.figure()
    for ni,N in enumerate(N_vals[1:2]):
        for ti in range(1,5):
            for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
                D_array = np.array(res["tmrca_mean"][ir, dr, N, :].index)
                label = f'r={ir}, a={dr}' if N==N_vals[0] else ''
                plt.errorbar(D_array*N, np.array([res["z_mean"].loc[(ir, dr, N, d)][ti] for d in D_array]),
                                        np.array([res["z_std"].loc[(ir, dr, N, d)][ti]/np.sqrt(res["nobs"][ir, dr, N]) for d in D_array]),
                    label=label, ls=ls[ti%len(ls)], c=f"C{i%10}")
                # plt.errorbar(D_array*N, np.array([res["z_mean"].loc[(ir, dr, N, d)][0:1] for d in D_array]).mean(axis=1)/res["diffusion_mean"][ir, dr, N, :]*D_array,
                #                         np.array([res["z_std"].loc[(ir, dr, N, d)]/np.sqrt(res["nobs"][ir, dr, N]) for d in D_array]).mean(axis=1),
                #      label=label, ls=ls[ni%len(ls)], c=f"C{i%10}")

    plt.legend()
    plt.plot(N*D_array, np.ones_like(D_array), c='k')
    plt.xlabel('diffusion constant $[L^2/N]$')
    plt.ylabel('Coverage')
    # plt.yscale('log')
    plt.xscale('log')
    if args.output_zscore:
        plt.savefig(args.output_zscore)

    # Figure showing the TMRCA of the population
    plt.figure()
    for ni, N in enumerate(N_vals[:]):
        for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
            D_array = np.array(res["tmrca_mean"][ir, dr, N, :].index)
            label = f'r={ir}' if N==N_vals[1] else ''
            plt.errorbar(D_array*N, res["tmrca_mean"][ir, dr, N, :]/N/2,
                                    res["tmrca_std"][ir, dr, N, :]/N/2/np.sqrt(res["nobs"][ir, dr, N]),
                 label=label, ls=ls[ni%len(ls)], c=f"C{i%10}")

    plt.legend()
    plt.plot(N*D_array, np.ones_like(D_array), c='k')
    plt.xlabel('diffusion constant $[L^2/N]$')
    plt.ylabel('$T_mrca/2N$')
    # plt.yscale('log')
    plt.xscale('log')
    if args.output_tmrca:
        plt.savefig(args.output_tmrca)

    # Figure showing the TMRCA of the population
    plt.figure()
    for N in N_vals[1:]:
        for i, (ir, dr) in enumerate(product(ir_to_plot, density_reg_to_plot)):
            D_array = np.array(res["tmrca_mean"][ir, dr, N, :].index)
            label = f'r={ir}' if N==N_vals[1] else ''
            for ti in range(4):
                plt.plot(D_array*N, [res["x_err_abs"].loc[ir, dr, N, Dval][ti] for Dval in D_array],
                     label=label, ls=ls[i%len(ls)], c=f"C{i//len(ls)}")

    plt.legend()
    plt.xlabel('diffusion constant $[L^2/N]$')
    plt.ylabel('Error')
    # plt.yscale('log')
    plt.xscale('log')
    if args.output_tmrca:
        plt.savefig(args.output_tmrca)
