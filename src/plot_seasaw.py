import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from heterogeneity import get_granularity
from itertools import product
from parse_and_plot import parse_data, make_figure, add_panel_label
from habitat_shifts import seasaw

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
    parser.add_argument('--data', type=str, default='data/seasaw.csv')
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    Lx, Ly = 3, 1
    fs=12

    make_figure(seasaw, {'period':8, 'amplitude':1.1}, fname="figures/seasaw_illustration.pdf", panel_label='A')

    res = parse_data(data, groupby=['interaction_radius', 'density_reg', 'N', 'period', 'subsampling'])
    interaction_radius = data.interaction_radius.unique()
    period = data.period.unique()
    subsampling = data.subsampling.unique()
    density_reg = data.density_reg.unique()
    N_vals = data.N.unique()

    fig, axs = plt.subplots(1,2, figsize=(12,3))

    ## Diffusion constant estimates
    ir = interaction_radius[1]
    dr = density_reg[2]
    p = 1.0
    ax=axs[0]
    for ti,T in enumerate([200, 500]):
        for m, N in zip(['o', '<'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            x_vals = D_array*N/Ly**2
            ax.errorbar(x_vals, res['diffusion_mean_x'][ir, dr, N, T, p, :]/D_array,
                                res['diffusion_std_x'][ir, dr, N, T, p, :]/D_array/np.sqrt(res['nobs'][ir, dr, N, T, p]), marker=m,
                                label=f'N={N}, T={T}', ls='-', c=f"C{ti%10}")
            ax.errorbar(x_vals, res['diffusion_mean_y'][ir, dr, N, T, p, :]/D_array,
                                res['diffusion_std_y'][ir, dr, N, T, p, :]/D_array/np.sqrt(res['nobs'][ir, dr, N, T, p]), marker=m,
                                label=f'', ls='--', c=f"C{ti%10}")
                                #label=f'D_y; N={N}, T={T}', ls='--', c=f"C{ti%10}")

    ax.set_xlabel(r'diffusion constant $[L_y^2/T_c]$', fontsize=fs)
    ax.set_ylabel(r'$\hat{D} / D$', fontsize=fs)
    #ax.set_yscale('log')
    ax.set_ylim(0,2)
    ax.set_xscale('log')
    #ax.colorbar()
    ax.legend()
    ax.plot(ax.get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)
    add_panel_label(ax, 'D')

    ## Diffusion constant estimates
    ax=axs[1]
    for ti,T in enumerate([200, 500]):
        for m, N in zip(['o', '<'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            x_vals = D_array*N/Ly**2
            plt.errorbar(D_array*N, np.array([res["z_mean_x"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,:1].mean(axis=1),
                            np.array([res["z_std_x"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,:2].mean(axis=1)/np.sqrt(res["nobs"][ir, dr, N, T, p]),
                label=f'N={N}, T={T}', ls='-', c=f"C{ti%10}", marker=m)
            plt.errorbar(D_array*N, np.array([res["z_mean_y"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,:1].mean(axis=1),
                            np.array([res["z_std_y"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,:2].mean(axis=1)/np.sqrt(res["nobs"][ir, dr, N, T, p]),
                label=f'', ls='--', c=f"C{ti%10}", marker=m)
                #label=f'N={N}, T={T}', ls='--', c=f"C{ti%10}", marker=m)

    ax.set_ylabel(r'$(x - \hat{x})^2/2\sigma^2$', fontsize=fs)
    ax.set_xlabel(r'diffusion constant $[L_y^2/T_c]$', fontsize=fs)
    #ax.set_yscale('log')
    ax.set_ylim(0)
    ax.set_xscale('log')
    #ax.colorbar()
    # ax.legend()
    ax.plot(ax.get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)
    add_panel_label(ax, 'E')

    plt.tight_layout()
    plt.savefig("figures/seasaw.pdf")