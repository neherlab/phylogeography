import numpy as np
from free_diffusion import *
from estimate_diffusion_from_tree import *
from itertools import product

if __name__=="__main__":
    replicates = 10
    results = {}
    fields = ['Dx_branch', 'Dy_branch', 'vx_branch', 'vy_branch', 'Dx_total', 'Dy_total', 'vx_total', 'vy_total']
    for n, yule in product([10,30, 100, 300, 1000], [True, False]):
        res = []
        for i in range(replicates):
            tree = random_tree(n, yule=yule)['tree']
            add_positions(tree, 1)
            clean_tree(tree)
            D = estimate_diffusion(tree)
            res.append(D)

        results[n] = {k: np.mean([x[k] for x in res]) for k in fields}
        results[n].update({f"{k}_std": np.std([x[k] for x in res]) for k in fields})


    import matplotlib.pyplot as plt

    for suffix in ['_total', '_branch']:
        plt.figure()
        for p in ['Dx', 'vx']:
            q = p+suffix
            label = f"{q} [{'mean(dx_i^2)/mean(2t_i)' if 'D' in q else 'mean(dx_i)/mean(t_i)'}]" if suffix=='_total' else f"{q} [{'mean(dx_i^2/2t_i)' if 'D' in q else 'mean(dx_i/t_i)'}]"
            plt.errorbar([n for n in results], [results[n][q] for n in results], [results[n][q+"_std"] for n in results], ls = '-', marker='o',
                        label=label)

        plt.xlabel('Number of leaves')
        plt.ylabel('estimate')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(f'figures/{"yule" if yule else "kingman"}{suffix}_dispersal.png')
