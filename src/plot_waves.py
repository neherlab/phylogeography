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
    dgb = data.groupby(groupby+['D'])

    density_variation = dgb['density_variation'].mean()
    nobs = data.groupby(groupby)['n'].mean()/25
    diffusion_mean_x = dgb['meanDx'].mean()
    diffusion_mean_y = dgb['meanDy'].mean()
    v_mean_x = dgb['meanvx'].mean()
    v_mean_y = dgb['meanvy'].mean()
    diffusion_std_x = dgb['stdD_x'].mean()
    diffusion_std_y = dgb['stdD_y'].mean()
    tmrca_mean = dgb['meanTmrca'].mean()
    tmrca_std = dgb['stdTmrca'].mean()

    x_err = {}
    for g, d in dgb['x_err']:
        x_err[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])[:5]), axis=0)

    y_err = {}
    for g, d in dgb['y_err']:
        y_err[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])[:5]), axis=0)

    x_err_abs = {}
    for g, d in dgb['x_err']:
        x_err_abs[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()][5:])), axis=0)

    y_err_abs = {}
    for g, d in dgb['y_err']:
        y_err_abs[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])[5:]), axis=0)

    z_mean = {}
    for g, d in dgb['meanZsq_x']:
        z_mean[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    z_std = {}
    for g, d in dgb['stdZsq_x']:
        z_std[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    return {"density_variation": density_variation, "nobs":nobs,
            "diffusion_mean_x":diffusion_mean_x, "diffusion_std_x":diffusion_std_x,
            "diffusion_mean_y":diffusion_mean_y, "diffusion_std_y":diffusion_std_y,
            "v_mean_x":v_mean_x, "v_mean_y":v_mean_y,
            "x_err": pd.DataFrame(x_err).T, "y_err": pd.DataFrame(y_err).T,
            "x_err_abs": pd.DataFrame(x_err_abs).T, "y_err_abs": pd.DataFrame(y_err_abs).T,
            "tmrca_mean":tmrca_mean, "tmrca_std":tmrca_std,
            "z_mean": pd.DataFrame(z_mean).T, "z_std":pd.DataFrame(z_std).T,
            }




def make_figure(fname=None, width=0.5):
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(4,2))
    f = waves(1, 3, 1, velocity=1, width=width)
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
    parser.add_argument('--output-velocity', type=str)
    parser.add_argument('--output-zscore', type=str)
    parser.add_argument('--output-tmrca', type=str)
    parser.add_argument('--illustration', type=str)
    args = parser.parse_args()

    # make an illustration of the carrying capacity at different times
    make_figure(fname = args.illustration, width=0.5)


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


    ## Diffusion constant estimates
    for c in ['x']:
        plt.figure(figsize=(6,3))
        #plt.title(f"{T}")
        for vi,v in enumerate(velocity):
            labeled=False
            for m, N in zip(['o', '<'], N_vals):
                D_array = data.loc[data.N==N].D.unique()
                for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg[:2], [1.0])):
                    v_array = 2*np.sqrt(dr*D_array)
                    try:
                        plt.plot(v_array/v, res["diffusion_mean_"+c][ir, dr, N, v, p,:]/D_array**1.0,
                                            c=f'C{vi}', marker=markers[density_reg.index(dr)], ls='-',
                            label='' if labeled else f'v={v}')
                        labeled=True
                    except:
                        pass

            plt.ylabel(r'$\hat{D}_x / D$')
            plt.xlabel(r'$\frac{2\sqrt{\alpha D}}{v} = \frac{v_{FKPP}}{v}$')
            plt.yscale('log')
            plt.xscale('log')
        #plt.colorbar()
        plt.legend()
        plt.plot(plt.gca().get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)
        plt.tight_layout()
    if args.output_diffusion:
        plt.savefig(args.output_diffusion)


    ## velocity constant estimates
    for c in ['x']:
        plt.figure(figsize=(6,3))
        #plt.title(f"{T}")
        for vi,v in enumerate(velocity):
            labeled = False
            for m, N in zip(['o', '<'], N_vals):
                D_array = data.loc[data.N==N].D.unique()
                for i, (ir, dr, p) in enumerate(product(interaction_radius, density_reg[:2], [1])):
                    v_array = 2*np.sqrt(dr*D_array)
                    try:
                        plt.scatter(v_array/v, res["v_mean_"+c][ir, dr, N, v, p,:]/v,
                                            c=f'C{vi}', marker=m,
                            label='' if labeled else f'v={v}')
                        labeled=True
                    except:
                        pass

            plt.xlabel(r'$\frac{2\sqrt{\alpha D}}{v} = \frac{v_{FKPP}}{v}$')
            plt.ylabel(r'$\hat{v}_x / v$')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
        #plt.colorbar()
        #plt.legend()
        plt.plot(plt.gca().get_xlim(), [1,1], ls='-', c='k', lw=3, alpha=0.3)
        plt.tight_layout()
        if args.output_velocity:
            plt.savefig(args.output_velocity)



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