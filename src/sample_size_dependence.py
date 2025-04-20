import numpy as np
from free_diffusion import random_tree, add_positions
from estimate_diffusion_from_tree import estimate_diffusion

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
                add_positions(tree, 1, gauss=False, exponent=2.5, generation_time=0.1)
                D = estimate_diffusion(tree)
                res.append(D)
            print(n, np.mean([x['t_total'] for x in res]))
            results[coal][n] = {k: np.mean([x[k] for x in res]) for k in fields}
            results[coal][n].update({f"{k}_std": np.std([x[k] for x in res]) for k in fields})


    import matplotlib.pyplot as plt

    fs=12
    fig, axs = plt.subplots(1,2,figsize=(10,4), sharey=True, sharex=True)
    for ax, coal in zip(axs, ['yule', 'kingman']):
        ax.text(10,100, "Kingman coalescent" if coal=='kingman' else "Yule tree", fontsize=fs)
        res = results[coal]
        offset = 1.0/1.02
        for suffix, suffix_label in [('_total', 'weighted'), ('_branch', 'by branch')]:
            for p, quantity in [('Dxy', 'diffusion'), ('vxy', 'velocity')]:
                q = f"{p}{suffix}"
                label = f"{quantity}, {suffix_label}"
                ax.errorbar([n*offset for n in res], [res[n][q] for n in res],
                             [res[n][q+"_std"] for n in res],
                             ls = '-', marker='o', label=label)
                offset *= 1.02**2
        ax.plot([10, 3000], [1,1], ls='--', color='k')
        ax.set_xlabel('number of samples', fontsize=fs)
        if coal=='kingman':
            ax.plot([10, 3000], [10,10*300**0.5], ls='-.', color='k')
        if coal=='yule':
            ax.set_ylabel('estimate', fontsize=fs)
            ax.legend()
        ax.set_ylim(0.5, 200)
        ax.set_xlim(7, 3600)
        ax.set_yscale('log')
        ax.set_xscale('log')
    plt.tight_layout()

    plt.savefig('figures/dispersal_stats.pdf')
