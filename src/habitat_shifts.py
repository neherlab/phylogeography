import numpy as np
from density_regulation import make_node, evolve, clean_tree, subsample_tree
from heterogeneity import get_2d_hist
from density_regulation import subsample_tree
from estimate_diffusion_from_tree import estimate_diffusion, estimate_ancestral_positions, collect_errors

def cycling_patches(N, Lx, Ly, period, wave_length):
    def f(x,y,t):
        return N*(1+np.sin(2*np.pi*(x/wave_length/Lx + t/period))*np.cos(2*np.pi*(y/wave_length/Ly + 0.5*t/period)))
    return f


def waves(N, Lx, Ly, velocity, width):
    def f(x,y,t):
        pos = (Lx/2+velocity*t)%Lx 
        return N*(np.exp(-0.5*(np.minimum((pos - x%Lx)%Lx, (x%Lx - pos)%Lx)**2/width**2)))
    return f

def breathing(N, Lx, Ly, period, width):
    def f(x,y,t):
        pos = (Lx/2)%Lx 
        prefactor = 0.25*(1+np.cos(2*np.pi*t/period))/width**2
        return N*(np.exp(-prefactor*(min((pos - x%Lx)%Lx, (x%Lx - pos)%Lx)**2)))
    return f

def seasaw(N, Lx, Ly, period):
    def f(x,y,t):
        return N*np.maximum(0.01,np.minimum(1,0.5*(1+0.8*np.cos(2*np.pi*t/period)*np.cos(2*np.pi*x/Lx)**1)))
    return f

def diffusion_in_changing_habitats(D, interaction_radius, density_reg, N, subsampling=1.0,
                                Lx=1, Ly=1, linear_bins=5, n_iter=10, n_subsamples=1, gtd=None, habitat_params=None):
    # set up tree and initial population uniformly in space
    tree = make_node(Lx/2,Ly/2,-2, None)
    tree['children'] = [make_node(np.random.random()*Lx, np.random.random()*Ly, -1, tree)
                        for i in range(N)]
    terminal_nodes = tree['children']

    density_variation = []
    D_est_x = []
    D_est_y = []
    v_est_x = []
    v_est_y = []
    zscores_x = []
    zscores_y = []
    x_err = []
    y_err = []
    Tmrca = []

    for t in range((n_iter+10)*N):
        target_density = gtd(N, Lx, Ly, **habitat_params)
        terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly, interaction_radius=interaction_radius,
                                density_reg=density_reg, D=D, target_density=target_density, 
                                total_population=N)
        if len(terminal_nodes)<10:
            print("population nearly extinct")
            continue
        if t%(N//5)==0 and t>10*N: # take samples after burnin every Tc//5
            tbins = sorted([0] + [t - i*N/10 for i in range(4)])
            clean_tree(tree)
            H, bx, by = get_2d_hist(terminal_nodes, Lx, Ly, linear_bins)
            density_variation.append(np.std(H)/N*np.prod(H.shape))
            for sample in range(n_subsamples):
                subsample_tree(terminal_nodes, tree, p=subsampling, subtree_attr='clades')
                D_res = estimate_diffusion(tree)
                estimate_ancestral_positions(tree, D)
                z = collect_errors(tree)
                if len(tree['clades'])==1:
                    Tmrca.append(t-tree['clades'][0]['time'])
                else:
                    Tmrca.append(t)
                D_est_x.append(D_res['Dx_total'])
                D_est_y.append(D_res['Dy_total'])
                v_est_x.append(D_res['vx_total'])
                v_est_y.append(D_res['vy_total'])
                zscores_x.append([np.mean(z.loc[(z.t >= tbins[i]) & (z.t<tbins[i+1]), 'zx']**2) for i in range(len(tbins)-1)])
                zscores_y.append([np.mean(z.loc[(z.t >= tbins[i]) & (z.t<tbins[i+1]), 'zy']**2) for i in range(len(tbins)-1)])
                x_err.append([np.mean(z.loc[(z.t >= tbins[i]) & (z.t<tbins[i+1]), 'x_err']) for i in range(len(tbins)-1)] + 
                             [np.mean(np.abs(z.loc[(z.t >= tbins[i]) & (z.t<tbins[i+1]), 'x_err'])) for i in range(len(tbins)-1)])
                y_err.append([np.mean(z.loc[(z.t >= tbins[i]) & (z.t<tbins[i+1]), 'y_err']) for i in range(len(tbins)-1)] + 
                             [np.mean(np.abs(z.loc[(z.t >= tbins[i]) & (z.t<tbins[i+1]), 'y_err'])) for i in range(len(tbins)-1)])

    return {"density_variation": density_variation, "D_est_x": D_est_x, "D_est_y": D_est_y, "D_est": D_est_x + D_est_y,
            "v_est_x": v_est_x, "v_est_y": v_est_y, "v_est": v_est_x + v_est_y,
            'zscores_x':zscores_x, 'zscores_y':zscores_y,'x_err':x_err, 'y_err':y_err, "z_scores": zscores_x+zscores_y,  "Tmrca":Tmrca}

def test_density(Lx, Ly, tmax, gtd=None, habitat_params=None):
    d = gtd(1, Lx, Ly, **habitat_params)
    import matplotlib.pyplot as plt
    x_points = np.linspace(0,Lx,int(20*Lx))
    y_points = np.linspace(0,Ly,int(20*Ly))
    for i in np.linspace(0, tmax, 9):
        plt.matshow([[d(x,y,i) for x in x_points] for y in y_points], vmin=0, vmax=1)
    plt.show()


if __name__=="__main__":
    import sys
    sys.setrecursionlimit(10000)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--D', type=float, default=0.1)
    parser.add_argument('--interaction-radius', type=float, default=0.1)
    parser.add_argument('--density-reg', type=float, default=0.1)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--subsampling', type=float, default=1.0)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    N = args.N
    Lx, Ly = 1, 1
    res_density = {}
    res_density_mean = {}
    D_est = []
    D_array_dens = np.logspace(-3,0,21)*Lx*Ly*2/N
    n_iter = 50
    linear_bins=5
    interaction_radius, density_reg = args.interaction_radius, args.density_reg
    print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")
    for di, D in enumerate(D_array_dens):
        print(f"{di} out of {len(D_array_dens)}: D={D:1.3e}")
        res = diffusion_in_changing_habitats(D, interaction_radius, density_reg, N, subsampling=args.subsampling,
                                          Lx=Lx, Ly=Ly, linear_bins=linear_bins, n_iter=n_iter, 
                                          gtd=cycling_patches, habitat_params={'period':args.period, 'wave_length':1.0})

        tmpD = np.mean(res["D_est"], axis=0)
        tmpStdD = np.std(res["D_est"], axis=0)
        tmpZ =    f"[{' '.join(str(x) for x in np.ma.mean(np.ma.masked_invalid(res['zscores']), axis=0).filled(fill_value=np.nan))}]"
        tmpStdZ = f"[{' '.join(str(x) for x in np.ma.std(np.ma.masked_invalid(res['zscores']), axis=0).filled(fill_value=np.nan))}]"
        nobs = len(res["D_est"])
        D_est.append({"interaction_radius":interaction_radius, "density_reg": density_reg,
                      "N": N, "n": len(res["D_est"]), "period": args.period, "subsampling": args.subsampling,
                      "D":D, "meanD": tmpD, "stdD": tmpStdD,
                      "meanZsq": tmpZ, "stdZsq": tmpStdZ, 'observations': nobs,
                      "density_variation": np.mean(res['density_variation']),
                      "meanTmrca":np.mean(res["Tmrca"]), "stdTmrca":np.std(res["Tmrca"])})

    import pandas as pd
    if args.output:
        fname = args.output
    else:
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        fname = f'data/habitats_diffusion_{N=}_ir={interaction_radius}_dr={density_reg}.csv'

    pd.DataFrame(D_est).to_csv(fname, index=False)

