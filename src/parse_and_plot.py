import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True
})


# function that adds an single letter panel label to an ax object
def add_panel_label(ax, label, x=-0.1, y=1.1, fontsize=12):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='right')


def parse_data(data, groupby=None):
    dgb = data.groupby(groupby+['D'])

    density_variation = dgb['density_variation'].mean()
    nobs = data.groupby(groupby)['observations'].mean()/25
    symmetric = False
    try:
        diffusion_mean_x = dgb['meanDx'].mean()
        diffusion_mean_y = dgb['meanDy'].mean()
        v_mean_x = dgb['meanvx'].mean()
        v_mean_y = dgb['meanvy'].mean()
        diffusion_std_x = dgb['stdD_x'].mean()
        diffusion_std_y = dgb['stdD_y'].mean()
    except:
        symmetric = True
        diffusion_mean = dgb['meanD'].mean()
        diffusion_std = dgb['stdD'].mean()

    tmrca_mean = dgb['meanTmrca'].mean()
    tmrca_std = dgb['stdTmrca'].mean()

    x_err = {}
    for g, d in dgb['x_err']:
        x_err[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    y_err = {}
    for g, d in dgb['y_err']:
        y_err[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    try:
        x_err_abs = {}
        for g, d in dgb['x_err_abs']:
            x_err_abs[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

        y_err_abs = {}
        for g, d in dgb['y_err_abs']:
            y_err_abs[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

        x_err_sq = {}
        for g, d in dgb['x_err_sq']:
            x_err_sq[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

        y_err_sq = {}
        for g, d in dgb['y_err_sq']:
            y_err_sq[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)
    except:
        print("failed to parse error information")
        x_err_abs = pd.DataFrame()
        y_err_abs = pd.DataFrame()
        x_err_sq = pd.DataFrame()
        y_err_sq = pd.DataFrame()


    z_mean_x = {}
    for g, d in dgb['meanZsq_x']:
        z_mean_x[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    z_std_x = {}
    for g, d in dgb['stdZsq_x']:
        z_std_x[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    z_mean_y = {}
    for g, d in dgb['meanZsq_y']:
        z_mean_y[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    z_std_y = {}
    for g, d in dgb['stdZsq_y']:
        z_std_y[g] = np.mean(d.apply(lambda x:np.array([float(y) for y in x[1:-1].split()])), axis=0)

    if symmetric:
        return {"density_variation": density_variation, "nobs":nobs,
                "diffusion_mean":diffusion_mean, "diffusion_std":diffusion_std,
                "x_err": pd.DataFrame(x_err).T, "y_err": pd.DataFrame(y_err).T,
                "x_err_abs": pd.DataFrame(x_err_abs).T, "y_err_abs": pd.DataFrame(y_err_abs).T,
                "x_err_sq": pd.DataFrame(x_err_sq).T, "y_err_sq": pd.DataFrame(y_err_sq).T,
                "tmrca_mean":tmrca_mean, "tmrca_std":tmrca_std,
                "z_mean": pd.DataFrame(z_mean_x).T, "z_std":pd.DataFrame(z_std_x).T,
                }
    else:
        return {"density_variation": density_variation, "nobs":nobs,
                "diffusion_mean_x":diffusion_mean_x, "diffusion_std_x":diffusion_std_x,
                "diffusion_mean_y":diffusion_mean_y, "diffusion_std_y":diffusion_std_y,
                "v_mean_x":v_mean_x, "v_mean_y":v_mean_y,
                "x_err": pd.DataFrame(x_err).T, "y_err": pd.DataFrame(y_err).T,
                "x_err_abs": pd.DataFrame(x_err_abs).T, "y_err_abs": pd.DataFrame(y_err_abs).T,
                "x_err_sq": pd.DataFrame(x_err_sq).T, "y_err_sq": pd.DataFrame(y_err_sq).T,
                "tmrca_mean":tmrca_mean, "tmrca_std":tmrca_std,
                "z_mean_x": pd.DataFrame(z_mean_x).T, "z_std_x":pd.DataFrame(z_std_x).T,
                "z_mean_y": pd.DataFrame(z_mean_y).T, "z_std_y":pd.DataFrame(z_std_y).T
                }


def make_figure(func, params, Lx=3, Ly=1, time_points=5, fname=None, panel_label=None):
    fig, axs = plt.subplots(1,time_points, sharex=True, sharey=True, figsize=(15,2))
    f = func(1, Lx, Ly, **params)
    for i, ax in enumerate(axs.flatten()):
        grid = np.meshgrid(np.linspace(0,Lx,30*Lx), np.linspace(0,Ly,30))
        vals = f(grid[0], grid[1], i)
        norm = np.sum(vals)/30/Lx/15
        ax.matshow(vals/norm, vmax=1, vmin=0)
        ax.set_axis_off()
        if panel_label and i==0:
            add_panel_label(ax, panel_label, x=-0.1, y=1.1)
    plt.tight_layout()
    if fname:
        plt.savefig(fname)

