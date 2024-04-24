import numpy as np
from habitat_shifts import seasaw, diffusion_in_changing_habitats


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
    parser.add_argument('--amplitude', type=int, default=1.1)
    parser.add_argument('--subsampling', type=float, default=1.0)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    N = args.N
    width = 0.5
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
                                          gtd=seasaw, habitat_params={'period':args.period, 'amplitude':1.1})

        tmpD_x = np.mean(res["D_est_x"], axis=0)
        tmpD_y = np.mean(res["D_est_y"], axis=0)
        tmpStdD_x = np.std(res["D_est_x"], axis=0)
        tmpStdD_y = np.std(res["D_est_y"], axis=0)
        tmpv_x = np.mean(res["v_est_x"], axis=0)
        tmpv_y = np.mean(res["v_est_y"], axis=0)
        tmpStdv_x = np.std(res["v_est_x"], axis=0)
        tmpStdv_y = np.std(res["v_est_y"], axis=0)
        tmpZ_x =    f"[{' '.join(str(x) for x in np.ma.mean(np.ma.masked_invalid(res['zscores_x']), axis=0).filled(fill_value=np.nan))}]"
        tmpStdZ_x = f"[{' '.join(str(x) for x in np.ma.std(np.ma.masked_invalid(res['zscores_x']), axis=0).filled(fill_value=np.nan))}]"
        tmpZ_y =    f"[{' '.join(str(x) for x in np.ma.mean(np.ma.masked_invalid(res['zscores_y']), axis=0).filled(fill_value=np.nan))}]"
        tmp_x_err =  f"[{' '.join(str(x) for x in np.ma.mean(np.ma.masked_invalid(res['x_err']), axis=0).filled(fill_value=np.nan))}]"
        tmp_y_err =  f"[{' '.join(str(x) for x in np.ma.mean(np.ma.masked_invalid(res['y_err']), axis=0).filled(fill_value=np.nan))}]"
        tmpStdZ_y = f"[{' '.join(str(x) for x in np.ma.std(np.ma.masked_invalid(res['zscores_y']), axis=0).filled(fill_value=np.nan))}]"
        nobs = len(res["D_est"])
        D_est.append({"interaction_radius":interaction_radius, "density_reg": density_reg,
                      "N": N, "n": len(res["D_est"]), "period": args.period, "subsampling": args.subsampling, "D":D, 
                      "meanDx": tmpD_x, "stdD_x": tmpStdD_x, "meanDy": tmpD_y, "stdD_y": tmpStdD_y,
                      "meanvx": tmpv_x, "stdv_x": tmpStdv_x, "meanvy": tmpv_y, "stdv_y": tmpStdv_y,
                      "meanZsq_x": tmpZ_x, "stdZsq_x": tmpStdZ_x, 
                      "meanZsq_y": tmpZ_y, "stdZsq_y": tmpStdZ_y, 
                      "x_err": tmp_x_err, "y_err": tmp_y_err, 
                      'observations': nobs,
                      "density_variation": np.mean(res['density_variation']),
                      "meanTmrca":np.mean(res["Tmrca"]), "stdTmrca":np.std(res["Tmrca"])})

    import pandas as pd
    if args.output:
        fname = args.output
    else:
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        fname = f'data/seasaw_{N=}_ir={interaction_radius}_dr={density_reg}.csv'

    pd.DataFrame(D_est).to_csv(fname, index=False)

