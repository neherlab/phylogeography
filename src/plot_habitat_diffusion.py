import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_inflated_diffusion import free_diffusion
from itertools import product
from habitat_shifts import generate_target_density

def make_figure(fname=None):
    fig, axs = plt.subplots(1,10, sharex=True, sharey=True, figsize=(15,2))
    f = generate_target_density(1, 1, 1, period=9, wave_length=1.0)
    for i, ax in enumerate(axs.flatten()):
        grid = np.meshgrid(np.linspace(0,1,30), np.linspace(0,1,30))
        ax.matshow(f(grid[0], grid[1], i))
        ax.set_axis_off()
    plt.tight_layout()
    if fname:
        plt.savefig(fname)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/habitat_diffusion.csv')
    parser.add_argument('--output-diffusion', type=str)
    parser.add_argument('--output-heterogeneity', type=str)
    parser.add_argument('--output-zscore', type=str)
    parser.add_argument('--output-tmrca', type=str)
    parser.add_argument('--illustration', type=str)
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    linear_bins=5
    Lx, Ly = 1, 1
    nbins=linear_bins**2

    density_variation = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['density_variation'].mean()
    nobs = data.groupby(['interaction_radius', 'density_reg', 'N', 'period', 'subsampling'])['n'].mean()/25
    diffusion_mean = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['meanD'].mean()
    diffusion_std = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['stdD'].mean()
    z_mean = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['meanZsq'].mean()
    z_std = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['stdZsq'].mean()
    tmrca_mean = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['meanTmrca'].mean()
    tmrca_std = data.groupby(['interaction_radius', 'density_reg', 'D', 'N', 'period', 'subsampling'])['stdTmrca'].mean()
    interaction_radius = data.interaction_radius.unique()
    period = data.period.unique()
    subsampling = data.subsampling.unique()
    density_reg = data.density_reg.unique()
    N_vals = data.N.unique()

    make_figure(fname = args.illustration)

    ls = ['-', ':', "--", ".-"][len(interaction_radius)]
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
    #plt.title(f"{T}")
    for T in period:
        for m, N in zip(['o', '<'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,
                                                    [1])):
                try:
                    plt.errorbar(np.sqrt(D_array*dr)*T/Lx, diffusion_mean[ir, dr, :, N, T, p]/D_array,
                                        diffusion_std[ir, dr, :, N, T, p]/D_array/np.sqrt(nobs[ir, dr, N, T, p]), marker=m,
                        label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")
                except:
                    pass

        #plt.plot(D_array/Lx/Ly, np.ones_like(D_array), c='k')
        plt.xlabel('sqrt(D a)T/L')
        plt.ylabel('estimated D / true D')
        plt.yscale('log')
        plt.xscale('log')
        #plt.legend()
    plt.plot(plt.gca().get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)

    for T in period:
        plt.figure()
        for m, N in zip(['o', 'd'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,
                                                    [1])):
                plt.errorbar(D_array*N, z_mean[ir, dr, :, N, T, p],
                                    z_std[ir, dr, :, N, T, p]/np.sqrt(nobs[ir, dr, N, T, p]),
                 label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)

        plt.legend()
        plt.plot(N*D_array, np.ones_like(D_array), c='k')
        plt.xlabel('true N*D')
        plt.ylabel('Coverage')
        plt.xscale('log')
    if args.output_zscore:
        plt.savefig(args.output_zscore)


    for T in period:
        plt.figure()
        #plt.title(f"{T}")
        for m, N in zip(['o', '<'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,
                                                    [1])):
                try:
                    plt.errorbar(D_array*N, tmrca_mean[ir, dr, :, N, T, p]/N/2,
                                            tmrca_std[ir, dr, :, N, T, p]/N/2/np.sqrt(nobs[ir, dr, N, T, p]),
                        label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)
                except:
                    pass

        plt.legend()
        plt.plot(N*D_array, np.ones_like(D_array), c='k')
        plt.xlabel('true N*D')
        plt.ylabel('T_mrca/2N')
        plt.xscale('log')
    if args.output_tmrca:
        plt.savefig(args.output_tmrca)
