import numpy as np
from density_regulation import run_simulation, serialize_result_isotropic

if __name__=="__main__":
    import sys
    sys.setrecursionlimit(10000)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--interaction-radius', type=float, default=0.1)
    parser.add_argument('--density-reg', type=float, default=0.1)
    parser.add_argument('--subsampling', type=float, default=1.0)
    parser.add_argument('--periodic', action='store_true', default=False)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    # Evolve a population with density regulation and estimate the diffusion constant, the TMRCA and the coverage of diffusion estimates

    N = args.N
    Lx, Ly = 1, 1
    res_density = {}
    res_density_mean = {}
    results = []
    D_array_dens = np.logspace(-3,0,21)*Lx*Ly*2/N
    n_iter = 50
    linear_bins=5
    interaction_radius, density_reg = args.interaction_radius, args.density_reg
    nsub = 1 if args.subsampling>0.9 else 5
    print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")
    for di, D in enumerate(D_array_dens):
        print(f"{di} out of {len(D_array_dens)}: D={D:1.3e}")
        res = run_simulation(D, interaction_radius, density_reg, N, subsampling=args.subsampling, periodic=args.periodic,
                             Lx=Lx, Ly=Ly, linear_bins=linear_bins, n_iter=n_iter, n_subsamples=nsub)
        processed_res = serialize_result_isotropic(res)
        processed_res.update({"interaction_radius":interaction_radius, "density_reg": density_reg,
                             "N": N,  "subsampling": args.subsampling, "D":D, "periodic":args.periodic})

        results.append(processed_res)

    import pandas as pd
    if args.output:
        fname = args.output
    else:
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        fname = f'data/stable_density_{N=}_ir={interaction_radius}_dr={density_reg}.csv'

    pd.DataFrame(results).to_csv(fname, index=False)

