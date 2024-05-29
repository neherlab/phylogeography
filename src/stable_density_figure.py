import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from heterogeneity import get_granularity
from itertools import product
from parse_and_plot import parse_data, add_panel_label

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
    parser.add_argument('--dataPBC', type=str, default='data/stable_density_periodicBC.csv')
    args = parser.parse_args()

    linear_bins=5
    Lx, Ly = 1, 1
    nbins=linear_bins**2

    data = pd.read_csv(args.data)
    res = parse_data(data, groupby=['interaction_radius', 'density_reg', 'N'])

    dataPBC = pd.read_csv(args.dataPBC)
    resPBC = parse_data(dataPBC, groupby=['interaction_radius', 'density_reg', 'N'])

    interaction_radius = data.interaction_radius.unique()
    density_reg = data.density_reg.unique()
    N_vals = data.N.unique()

    # figure
    fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

    # Plot the heterogeneity
    ls = ['-', '-.', "--", ":"] #[:len(density_reg_to_plot)]
    N=500
    dr = density_reg[1]
    ax=axs[0]
    for i, ir in enumerate(interaction_radius[1:]):
        D_array = np.array(resPBC["density_variation"][ir, dr, N, :].index)
        label = f'r={ir}' if N==N_vals[1] else ''
        ax.plot(D_array*N/Lx/Ly, resPBC["density_variation"][ir, dr, N, :]/np.sqrt(nbins/N),
                label=label, ls='-', c=f"C{i%10}")
    free_diffusion_heterogeneity = free_diffusion(D_array*N/Lx/Ly, N, linear_bins=linear_bins)
    ax.plot(D_array*N/Lx/Ly, free_diffusion_heterogeneity/np.sqrt(nbins/N), label='free diffusion', c='k')
    ax.plot(D_array*N/Lx/Ly, np.ones_like(D_array), c='k', ls='--')
    ax.set_xscale('log')
#    ax.set_xlabel('diffusion constant $[L^2/N]$')
    ax.set_ylabel('rel. heterogeneity')
    ax.legend()
    ax.set_xlim(0)
    add_panel_label(ax, 'C')
    #ax.text(f'N={N}, alpha={dr}, periodic boundary conditions')


    ## Plot the diffusion estiamtes
    ax=axs[1]
    for i, ir in enumerate(interaction_radius[1:]):
        D_array = np.array(res["diffusion_mean"][ir, dr, N, :].index)
        label = f'r={ir}, a={dr}' if N==N_vals[0] else ''
        ax.errorbar(D_array*N/Lx/Ly, res["diffusion_mean"][ir, dr, N, :]/D_array,
                                res["diffusion_std"][ir, dr, N, :]/D_array/np.sqrt(res["nobs"][ir, dr, N]), marker='o', ls='-',
                label=label, c=f"C{i%10}")
        ax.errorbar(D_array*N/Lx/Ly, resPBC["diffusion_mean"][ir, dr, N, :]/D_array,
                                resPBC["diffusion_std"][ir, dr, N, :]/D_array/np.sqrt(res["nobs"][ir, dr, N]), marker='v', ls='--',
                label=label,  c=f"C{i%10}")

    ax.plot(N*D_array/Lx/Ly, np.ones_like(D_array), c='k')
#    ax.set_xlabel('true $ND/L^2$')
    ax.set_ylabel('fold-error in D estimate')
    # plt.yscale('log')
    ax.set_xscale('log')
    add_panel_label(ax, 'D')

    # # plot the z-scores, i.e. the degree to which the estimates cover the true value
    # for ti in range(1,5):
    #     for i, ir in enumerate(interaction_radius[1:]):
    #         D_array = np.array(res["tmrca_mean"][ir, dr, N, :].index)
    #         label = f'r={ir}, a={dr}' if N==N_vals[0] else ''
    #         axs[2].errorbar(D_array*N, np.array([res["z_mean"].loc[(ir, dr, N, d)][ti] for d in D_array]),
    #                     np.array([res["z_std"].loc[(ir, dr, N, d)][ti]/np.sqrt(res["nobs"][ir, dr, N]) for d in D_array]),
    #                     label=label, c=f"C{i%10}")
    #             # plt.errorbar(D_array*N, np.array([res["z_mean"].loc[(ir, dr, N, d)][0:1] for d in D_array]).mean(axis=1)/res["diffusion_mean"][ir, dr, N, :]*D_array,
    #             #                         np.array([res["z_std"].loc[(ir, dr, N, d)]/np.sqrt(res["nobs"][ir, dr, N]) for d in D_array]).mean(axis=1),
    #             #      label=label, ls=ls[ni%len(ls)], c=f"C{i%10}")

    # axs[2].plot(N*D_array, np.ones_like(D_array), c='k')
    # axs[2].set_xlabel('diffusion constant $[L^2/N]$')
    # axs[2].set_ylabel('Coverage')
    # # plt.yscale('log')
    # axs[2].set_xscale('log')
    # plt.legend()

    # Figure showing the TMRCA of the population
    ax=axs[2]
    for i, ir in enumerate(interaction_radius[1:]):
        D_array = np.array(res["tmrca_mean"][ir, dr, N, :].index)
        label = f'r={ir}' if N==N_vals[1] else ''
        ax.errorbar(D_array*N, res["tmrca_mean"][ir, dr, N, :]/N/2,
                    res["tmrca_std"][ir, dr, N, :]/N/2/np.sqrt(res["nobs"][ir, dr, N]),
                     label=label, c=f"C{i%10}")

    ax.plot(N*D_array, np.ones_like(D_array), c='k')
    ax.set_xlabel('diffusion constant $[L^2/N]$')
    ax.set_ylabel('$T_mrca/2N$')
    ax.set_xscale('log')
    add_panel_label(ax, 'E')
    # plt.yscale('log')
    #plt.legend()

    # # Figure showing the TMRCA of the population
    # ax=axs[3]
    # for i, ir in enumerate(interaction_radius[2:3]):
    #     D_array = np.array(res["tmrca_mean"][ir, dr, N, :].index)
    #     label = f'r={ir}' if N==N_vals[1] else ''
    #     for ti in range(4):
    #         ax.plot(D_array*N, [np.sqrt(res["x_err_sq"].loc[ir, dr, N, Dval][ti]) for Dval in D_array], ls='-',
    #                 label=label, c=f"C{ti%10}")
    #         ax.plot(D_array*N, [np.sqrt(resPBC["x_err_sq"].loc[ir, dr, N, Dval][ti]) for Dval in D_array], ls='--',
    #                 label=label, c=f"C{ti%10}")

    # ax.set_xlabel('diffusion constant $[L^2/N]$')
    # ax.set_ylabel('Error')
    # # ax.set_yscale('log')
    # ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('figures/stable_density.pdf')
