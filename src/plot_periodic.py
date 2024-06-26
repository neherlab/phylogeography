import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_stable_density import free_diffusion
from parse_and_plot import parse_data, make_figure
from itertools import product
from habitat_shifts import cycling_patches,breathing, seasaw

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
    Lx, Ly = 3, 1
    nbins=linear_bins**2

    # make an illustration of the carrying capacity at different times
    make_figure(seasaw, {'period':8, 'amplitude':1.1}, fname=args.illustration)
    # make an illustration of the carrying capacity at different times
    make_figure(breathing, {'period':8, 'width':0.5}, fname=args.illustration)

    res = parse_data(data, groupby=['interaction_radius', 'density_reg', 'N', 'period', 'subsampling'])
    interaction_radius = data.interaction_radius.unique()
    period = data.period.unique()
    subsampling = data.subsampling.unique()
    density_reg = data.density_reg.unique()
    N_vals = data.N.unique()


    ls = ['-', ':', "--", ".-"][len(interaction_radius)]
    markers = ['o', '<', '>', 's', 'd', '^', 'v']

    # ## Population heterogeneity
    # plt.figure()
    # for N in N_vals:
    #     D_array = data.loc[data.N==N].D.unique()
    #     area = np.minimum(1,D_array*np.pi**2)
    #     for i, (ir, dr, T, p) in enumerate(product(interaction_radius, density_reg,
    #                                                period, subsampling)):
    #         try:
    #             plt.plot(D_array*N/Lx/Ly, res["density_variation"][ir, dr, N, T, p,:],
    #                     label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")
    #         except:
    #             pass

    #     free_diffusion_heterogeneity = free_diffusion(D_array*N/Lx/Ly, N, linear_bins=linear_bins)
    #     plt.plot(D_array*N/Lx/Ly, free_diffusion_heterogeneity, label='diffusion', c='k')
    #     plt.plot(D_array*N/Lx/Ly, np.ones_like(D_array)*np.sqrt(nbins/N), label='well mixed limit', c='k', ls='--')
    # plt.xscale('log')
    # plt.xlabel('diffusion constant $D$')
    # plt.ylabel('heterogeneity')
    # plt.legend()
    # if args.output_heterogeneity:
    #     plt.savefig(args.output_heterogeneity)


    ## Diffusion constant estimates
    for ti,T in enumerate(period[3:]):
        plt.figure()
        plt.title(f"{T}")
        for m, N in zip(['o', '<'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,[1])):
                x_vals = np.sqrt(D_array*dr)*T/Lx
                x_vals = D_array*N/Ly**2
                try:
                    # plt.errorbar(np.sqrt(D_array*dr)*T/Lx, diffusion_mean[ir, dr, :, N, T, p]/D_array,
                    #                     diffusion_std[ir, dr, :, N, T, p]/D_array/np.sqrt(nobs[ir, dr, N, T, p]), marker=m,
                    #     label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}")
                    plt.plot(x_vals, res["diffusion_mean_x"][ir, dr, N, T, p,:]/D_array**1.0, c=f'C{i%10}')
                    # plt.scatter(x_vals, res["diffusion_mean_x"][ir, dr, N, T, p,:]/D_array**1.0,
                    #                     c=res["tmrca_mean"][ir, dr, N, T, p,:]/2/N, marker=markers[i%len(markers)],
                    #     label=f'r={ir}, a={dr} N={N}, T={T} p={p}')
                    plt.plot(x_vals, res["diffusion_mean_y"][ir, dr, N, T, p,:]/D_array**1.0, c='k')
                except:
                    pass

        #plt.plot(D_array/Lx/Ly, np.ones_like(D_array), c='k')
        plt.xlabel(r'$DN/L_y^2$')
        plt.ylabel(r'$\hat{D} / D$')
        plt.yscale('log')
        plt.xscale('log')
    #plt.colorbar()
    #plt.legend()
    plt.plot(plt.gca().get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)

    for T in period[3:]:
        plt.figure()
        for m, N in zip(['o', 'd'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            try:
                for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,[1])):
                    plt.errorbar(D_array*N, np.array([res["z_mean"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,:2].mean(axis=1),
                                    np.array([res["z_std"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,:2].mean(axis=1)/np.sqrt(res["nobs"][ir, dr, N, T, p]),
                     label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)
            except:
                print(f"r={ir}, a={dr} N={N}, T={T} p={p}")

        plt.legend()
        plt.plot(N*D_array, np.ones_like(D_array), c='k')
        plt.xlabel('true N*D')
        plt.ylabel('Coverage')
        plt.xscale('log')
    if args.output_zscore:
        plt.savefig(args.output_zscore)


    for T in period:
        plt.figure()
        for m, N in zip(['o', 'd'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,[1])):
                try:
                    plt.plot(D_array*N, np.array([res["x_err_abs"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,0],
                    label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)
                    plt.plot(D_array*N, np.array([res["y_err_abs"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,0], c='k')
                except:
                    print(f"r={ir}, a={dr} N={N}, T={T} p={p}")

        plt.legend()
        plt.plot(N*D_array, np.zeros_like(D_array), c='k')
        plt.xlabel('true N*D')
        plt.ylabel('absolute error')
        plt.xscale('log')
    if args.output_zscore:
        plt.savefig(args.output_zscore)


    # for T in period:
    #     plt.figure()
    #     for m, N in zip(['o', 'd'], N_vals):
    #         D_array = data.loc[data.N==N].D.unique()
    #         for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,[1])):
    #             try:
    #                 plt.plot(D_array*N, np.array([res["y_err"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,1:-2].mean(axis=1),
    #                 label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)
    #             except:
    #                 print(f"r={ir}, a={dr} N={N}, T={T} p={p}")

    #     plt.legend()
    #     plt.plot(N*D_array, np.zeros_like(D_array), c='k')
    #     plt.xlabel('true N*D')
    #     plt.ylabel('error')
    #     plt.xscale('log')
    # if args.output_zscore:
    #     plt.savefig(args.output_zscore)

    for T in period:
        plt.figure()
        #plt.title(f"{T}")
        for m, N in zip(['o', '<'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,
                                                    [1])):
                try:
                    plt.errorbar(D_array*N, res["tmrca_mean"][ir, dr, N, T, p, :]/T,
                                            res["tmrca_std"][ir, dr, N, T, p, :]/T/np.sqrt(res["nobs"][ir, dr, N, T, p]),
                        label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)
                except:
                    pass

        plt.legend()
        plt.plot(N*D_array, np.ones_like(D_array), c='k')
        plt.xlabel(r'$N\times D$')
        plt.ylabel(r'$T_{mrca}/2N$')
        plt.xscale('log')
        plt.yscale('log')
    if args.output_tmrca:
        plt.savefig(args.output_tmrca)
