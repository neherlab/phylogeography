import numpy as np
from habitat_shifts import waves, test_density, diffusion_in_changing_habitats


if __name__=="__main__":
    import sys
    sys.setrecursionlimit(10000)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--D', type=float, default=0.1)
    parser.add_argument('--interaction-radius', type=float, default=0.1)
    parser.add_argument('--density-reg', type=float, default=0.1)
    parser.add_argument('--velocity', type=float, default=0.01)
    parser.add_argument('--subsampling', type=float, default=1.0)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    N = args.N
    width = 1
    Lx, Ly = 3, 1
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
                                          gtd=waves, habitat_params={'velocity':args.velocity, 'width':width})

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

