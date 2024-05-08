import numpy as np
from habitat_shifts import seasaw
from density_regulation import run_simulation, serialize_result


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
    Lx, Ly = 3, 1
    results = []
    D_array_dens = np.logspace(-3,0,21)*Lx*Ly*2/N
    n_iter = 50
    linear_bins=5
    interaction_radius, density_reg = args.interaction_radius, args.density_reg
    print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")
    for di, D in enumerate(D_array_dens):
        print(f"{di} out of {len(D_array_dens)}: D={D:1.3e}")
        res = run_simulation(D, interaction_radius, density_reg, N, subsampling=args.subsampling,
                                          Lx=Lx, Ly=Ly, linear_bins=linear_bins, n_iter=n_iter, periodic=False,
                                          gtd=seasaw, habitat_params={'period':args.period, 'amplitude':1.1})
        processed_res = serialize_result(res)
        processed_res.update({"interaction_radius":interaction_radius, "density_reg": density_reg,
                             "N": N, "period": args.period, "subsampling": args.subsampling, "D":D})

        results.append(processed_res)

    import pandas as pd
    if args.output:
        fname = args.output
    else:
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        fname = f'data/seasaw_{N=}_ir={interaction_radius}_dr={density_reg}.csv'

    pd.DataFrame(results).to_csv(fname, index=False)

