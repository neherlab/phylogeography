import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from habitat_shifts import waves
from parse_and_plot import parse_data, make_figure, add_panel_label


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/waves.csv')
    parser.add_argument('--output-diffusion', type=str)
    parser.add_argument('--output-velocity', type=str)
    parser.add_argument('--output-zscore', type=str)
    parser.add_argument('--output-tmrca', type=str)
    parser.add_argument('--illustration', type=str)
    args = parser.parse_args()

    # make an illustration of the carrying capacity at different times
    make_figure(func=waves, params={'width':0.5, 'velocity':1}, fname = args.illustration, panel_label='A')
    fs=12

    # read and organize data
    data = pd.read_csv(args.data)
    linear_bins=5
    Lx, Ly = 3, 1
    nbins=Lx*Ly*linear_bins**2


    res = parse_data(data, groupby=['interaction_radius', 'density_reg', 'N', 'velocity', 'subsampling'])
    velocity = sorted(data.velocity.unique())
    interaction_radius = sorted(data.interaction_radius.unique())
    subsampling = sorted(data.subsampling.unique())
    density_reg = sorted(data.density_reg.unique())
    N_vals = sorted(data.N.unique())

    ls = ['-', ':', "--", ".-"][len(interaction_radius)]
    markers = ['o', '<', '>', 's', 'd', '^', 'v']
    fig, axs = plt.subplots(2,1, figsize=(6,6), sharex=True)
    ax=axs[0]
    ## Diffusion constant estimates
    for c in ['x']:
        #plt.title(f"{T}")
        for vi,v in enumerate(velocity):
            labeled=False
            for m, N in zip(['o', '<'], N_vals):
                D_array = data.loc[data.N==N].D.unique()
                for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg[:2], [1.0])):
                    v_array = 2*np.sqrt(dr*D_array)
                    try:
                        ax.plot(v_array/v, res["diffusion_mean_"+c][ir, dr, N, v, p,:]/D_array**1.0,
                                            c=f'C{vi}', marker=markers[density_reg.index(dr)], ls='-',
                            label='' if labeled else f'v={v}')
                        labeled=True
                    except:
                        pass

            ax.set_ylabel(r'$\hat{D}_x / D$', fontsize=12)
            # ax.set_xlabel(r'$\frac{2\sqrt{\alpha D}}{v} = \frac{v_{FKPP}}{v}$')
            ax.set_yscale('log')
            ax.set_xscale('log')
        #plt.colorbar()
        ax.legend()
        ax.plot(plt.gca().get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)
    add_panel_label(ax, 'B')
    ## velocity constant estimates
    ax = axs[1]
    for c in ['x']:
        #plt.title(f"{T}")
        for vi,v in enumerate(velocity):
            labeled = False
            for m, N in zip(['o', '<'], N_vals):
                D_array = data.loc[data.N==N].D.unique()
                for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg[:2], [1])):
                    v_array = 2*np.sqrt(dr*D_array)
                    try:
                        ax.scatter(v_array/v, res["v_mean_"+c][ir, dr, N, v, p,:]/v,
                                            c=f'C{vi}', marker=m,
                            label='' if labeled else f'v={v}')
                        labeled=True
                    except:
                        pass

        ax.set_xlabel(r'$\frac{2\sqrt{\alpha D}}{v} = \frac{v_{FKPP}}{v}$', fontsize=14)
        ax.set_ylabel(r'$\hat{v}_x / v$', fontsize=12)
        ax.set_yscale('log')
        ax.set_xscale('log')
            #ax.legend()
        #plt.colorbar()
        #plt.legend()
        ax.plot(plt.gca().get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)

    add_panel_label(ax, 'C')
    plt.tight_layout()
    plt.savefig('figures/waves.pdf')
    ## ADDITIONAL FIGURES

    for v in velocity:
        plt.figure()
        for m, N in zip(['o', 'd'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            v_array = 2*np.sqrt(dr*D_array)
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,[1])):
                try:
                    plt.plot(v_array, np.array([res["x_err"].loc[(ir, dr, N, v, p, d)] for d in D_array])[:,0],
                    label=f'r={ir}, a={dr} N={N}, v={v} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)
                except:
                    print(f"r={ir}, a={dr} N={N}, v={v} p={p}")

        plt.legend()
        plt.plot(v_array, np.zeros_like(D_array), c='k')
        plt.axvline(v)
        plt.xlabel('v_{fkpp}')
        plt.ylabel('error')
        plt.xscale('log')
    if args.output_zscore:
        plt.savefig(args.output_zscore)


    for v in velocity:
        plt.figure()
        for m, N in zip(['o', 'd'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            v_array = 2*np.sqrt(dr*D_array)
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,[1])):
                try:
                    plt.plot(v_array, np.array([res["x_err"].loc[(ir, dr, N, v, p, d)] for d in D_array])[:,1:-2].mean(axis=1),
                    label=f'r={ir}, a={dr} N={N}, v={v} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)
                except:
                    print(f"r={ir}, a={dr} N={N}, v={v} p={p}")

        plt.legend()
        plt.plot(v_array, np.zeros_like(D_array), c='k')
        plt.axvline(v)
        plt.xlabel('v_{fkpp}')
        plt.ylabel('error')
        plt.xscale('log')
    if args.output_zscore:
        plt.savefig(args.output_zscore)