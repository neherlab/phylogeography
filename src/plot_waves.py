import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from habitat_shifts import waves

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def parse_data(data, groupby=None):
    density_variation = data.groupby(groupby + ['D'])['density_variation'].mean()
    nobs = data.groupby(groupby)['n'].mean()/25
    diffusion_mean_x = data.groupby(groupby + ['D'])['meanDx'].mean()
    diffusion_mean_y = data.groupby(groupby + ['D'])['meanDy'].mean()
    v_mean_x = data.groupby(groupby + ['D'])['meanvx'].mean()
    v_mean_y = data.groupby(groupby + ['D'])['meanvy'].mean()
    diffusion_std_x = data.groupby(groupby + ['D'])['stdD_x'].mean()
    diffusion_std_y = data.groupby(groupby + ['D'])['stdD_y'].mean()
    tmrca_mean = data.groupby(groupby + ['D'])['meanTmrca'].mean()
    tmrca_std = data.groupby(groupby + ['D'])['stdTmrca'].mean()

    z_mean = {}
    for g, d in data.groupby(groupby + ['D'])['meanZsq_x']:
        z_mean[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    z_std = {}
    for g, d in data.groupby(groupby + ['D'])['stdZsq_x']:
        z_std[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    return {"density_variation": density_variation, "nobs":nobs,
            "diffusion_mean_x":diffusion_mean_x, "diffusion_std_x":diffusion_std_x,
            "diffusion_mean_y":diffusion_mean_y, "diffusion_std_y":diffusion_std_y,
            "v_mean_x":v_mean_x, "v_mean_y":v_mean_y,
            "tmrca_mean":tmrca_mean, "tmrca_std":tmrca_std,
            "z_mean": pd.DataFrame(z_mean).T, "z_std":pd.DataFrame(z_std).T,
            }




def make_figure(fname=None):
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(4,2))
    f = waves(1, 3, 1, velocity=1, width=1.0)
    grid = np.meshgrid(np.linspace(0,3,31), np.linspace(0,1,11))
    ax.matshow(f(grid[0], grid[1], 0))
    ax.set_axis_off()
    ax.arrow(10.0, 7, 10.0, 0, width=0.3)
    ax.text(14.5, 6,"v",  fontsize=14)
    plt.tight_layout()
    if fname:
        plt.savefig(fname)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/waves.csv')
    parser.add_argument('--output-diffusion', type=str)
    parser.add_argument('--output-zscore', type=str)
    parser.add_argument('--output-tmrca', type=str)
    parser.add_argument('--illustration', type=str)
    args = parser.parse_args()

    # make an illustration of the carrying capacity at different times
    make_figure(fname = args.illustration)


    # read and organize data
    data = pd.read_csv(args.data)
    linear_bins=5
    Lx, Ly = 3, 1
    nbins=Lx*Ly*linear_bins**2


    res = parse_data(data, groupby=['interaction_radius', 'density_reg', 'N', 'velocity', 'subsampling'])
    interaction_radius = data.interaction_radius.unique()
    velocity = sorted(data.velocity.unique())
    subsampling = data.subsampling.unique()
    density_reg = data.density_reg.unique()
    N_vals = data.N.unique()

    ls = ['-', ':', "--", ".-"][len(interaction_radius)]
    markers = ['o', '<', '>', 's', 'd', '^', 'v']


    ## Diffusion constant estimates
    for c in ['x', 'y']:
        plt.figure()
        #plt.title(f"{T}")
        for vi,v in enumerate(velocity):
            for m, N in zip(['o', '<'], N_vals):
                D_array = data.loc[data.N==N].D.unique()
                for i, (ir, dr, p) in enumerate(product(interaction_radius[:1], density_reg[1:2], [1])):
                    try:
                        plt.scatter(D_array*N, res["diffusion_mean_"+c][ir, dr, N, v, p,:]/D_array**1.0,
                                            c=f'C{vi}', marker=m,
                            label=f'r={ir}, a={dr} N={N}, v={v} p={p}')
                    except:
                        pass

            plt.ylabel(r'$\hat{D} / D$')
            plt.yscale('log')
            plt.xscale('log')
        #plt.colorbar()
        plt.legend()
        plt.plot(plt.gca().get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)


    ## Diffusion constant estimates
    for c in ['x', 'y']:
        plt.figure()
        #plt.title(f"{T}")
        for vi,v in enumerate(velocity):
            for m, N in zip(['o', '<'], N_vals):
                D_array = data.loc[data.N==N].D.unique()
                for i, (ir, dr, p) in enumerate(product(interaction_radius[:1], density_reg[1:2], [1])):
                    try:
                        plt.scatter(D_array*N, res["v_mean_"+c][ir, dr, N, v, p,:],
                                            c=f'C{vi}', marker=m,
                            label=f'r={ir}, a={dr} N={N}, v={v} p={p}')
                    except:
                        pass

            plt.ylabel(r'$\hat{D} / D$')
            plt.yscale('log')
            plt.xscale('log')
        #plt.colorbar()
        plt.legend()
        plt.plot(plt.gca().get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)


    for T in period:
        plt.figure()
        for m, N in zip(['o', 'd'], N_vals):
            D_array = data.loc[data.N==N].D.unique()
            for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg,
                                                    [1])):
                plt.errorbar(D_array*N, np.array([res["z_mean"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,:-2].mean(axis=1),
                                    np.array([res["z_std"].loc[(ir, dr, N, T, p, d)] for d in D_array])[:,:-2].mean(axis=1)/np.sqrt(res["nobs"][ir, dr, N, T, p]),
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
                    plt.errorbar(D_array*N, res["tmrca_mean"][ir, dr, N, T, p, :]/N/2,
                                            res["tmrca_std"][ir, dr, N, T, p, :]/N/2/np.sqrt(res["nobs"][ir, dr, N, T, p]),
                        label=f'r={ir}, a={dr} N={N}, T={T} p={p}', ls=ls[i%len(ls)], c=f"C{i//len(ls)}", marker=m)
                except:
                    pass

        plt.legend()
        plt.plot(N*D_array, np.ones_like(D_array), c='k')
        plt.xlabel(r'$N\times D$')
        plt.ylabel(r'$T_{mrca}/2N$')
        plt.xscale('log')
    if args.output_tmrca:
        plt.savefig(args.output_tmrca)
