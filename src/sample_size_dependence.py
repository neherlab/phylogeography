import numpy as np
from free_diffusion import *
from estimate_diffusion_from_tree import *
from itertools import product

if __name__=="__main__":
    replicates = 100
    results = {}
    fields = ['Dxy_branch', 'Dxy_total', 'vxy_branch', 'vxy_total', 't_total']
    for coal in ['yule', 'kingman']:
        results[coal] = {}
        for n in [10,30, 100, 300, 1000, 3000]:
            res = []
            for i in range(replicates):
                tree = random_tree(n, yule=coal=='yule')['tree']
                add_positions(tree, 1)
                D = estimate_diffusion(tree)
                res.append(D)
            print(n, np.mean([x['t_total'] for x in res]))
            results[coal][n] = {k: np.mean([x[k] for x in res]) for k in fields}
            results[coal][n].update({f"{k}_std": np.std([x[k] for x in res]) for k in fields})


    import matplotlib.pyplot as plt

    for coal in ['yule', 'kingman']:
        res = results[coal]
        plt.figure()
        offset = 1.0/1.02
        for suffix, suffix_label in [('_total', 'weighted'), ('_branch', 'by branch')]:
            for p, quantity in [('Dxy', 'diffusion'), ('vxy', 'velocity')]:
                q = f"{p}{suffix}"
                label = f"{quantity}, {suffix_label}"
                plt.errorbar([n*offset for n in res], [res[n][q] for n in res], [res[n][q+"_std"] for n in res], ls = '-', marker='o',
                            label=label)
                offset *= 1.02**2
        plt.plot([10, 3000], [1,1], ls='--', color='k')
        plt.plot([10, 3000], [10,10*300**0.5], ls='-.', color='k')

        plt.xlabel('Number of leaves')
        plt.ylabel('estimate')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(0.5, 200)
        plt.xlim(7, 3600)
        plt.savefig(f'figures/{coal}_dispersal.pdf')
