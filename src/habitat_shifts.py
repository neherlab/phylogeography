import numpy as np
from density_regulation import run_simulation

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
        pos = 0
        prefactor = 0.25*(1+np.cos(2*np.pi*t/period))/width**2
        return N*(np.exp(-prefactor*(x-pos)**2))
    return f

def seasaw(N, Lx, Ly, period, amplitude=0.9):
    def f(x,y,t):
        return N*np.maximum(0.01,np.minimum(1,0.5*(0.1+amplitude*np.cos(2*np.pi*t/period)*np.cos(np.pi*x/Lx)**1)))
    return f

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
        res = run_simulation(D, interaction_radius, density_reg, N, subsampling=args.subsampling,
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

