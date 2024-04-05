import numpy as np
from habitat_shifts import waves, seasaw
from density_regulation import make_node, evolve, subsample_tree, dict_to_phylo_tree
from estimate_diffusion_from_tree import estimate_diffusion, estimate_ancestral_positions
import matplotlib.pyplot as plt
from Bio import Phylo
from matplotlib.cm import viridis_r 


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
    width = 0.5
    Lx, Ly = 3, 1
    D = args.D
    interaction_radius, density_reg = args.interaction_radius, args.density_reg
    #target_density = waves(N, Lx, Ly, **{'velocity':args.velocity, "width":width})
    period = 100
    target_density = seasaw(N, Lx, Ly, **{'period':period})

    v_FKPP = 2*np.sqrt(args.velocity*D)
    print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")

    tree = make_node(Lx/2,Ly/2,-2, None)
    tree['children'] = [make_node(np.random.random()*Lx, np.random.random()*Ly, -1, tree)
                        for i in range(N)]
    terminal_nodes = tree['children']

    t = 0
    for tmax in [5*N + i*period/3.0 for i in range(5)]:
        while t<tmax:
            terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly, interaction_radius=interaction_radius,
                                density_reg=density_reg, D=D, target_density=target_density, total_population=N)
            t+=1

        subsample_tree(terminal_nodes, tree, p=1.0, subtree_attr='clades')
        D_res = estimate_diffusion(tree)
        estimate_ancestral_positions(tree, D)
        phylo_tree = dict_to_phylo_tree(tree['clades'][0], child_attr='clades')

        fig, axs = plt.subplots(1,2, figsize=(12,8))
        Phylo.draw(phylo_tree, axes=axs[0], label_func=lambda x: '')

        for n in phylo_tree.get_nonterminals():
            axs[1].plot([n.pos['x'], n.inferred_pos['x']['mean']], [n.pos['y'], n.inferred_pos['y']['mean']], c='k', lw=0.5, alpha=0.5)
            for c in n.clades:
                axs[1].plot([n.pos['x'], c.pos['x']], [n.pos['y'], c.pos['y']], c='k', lw=0.5, alpha=0.5)

        axs[1].errorbar([n.inferred_pos['x']['mean'] for n in phylo_tree.find_clades()], [n.inferred_pos['y']['mean'] for n in phylo_tree.find_clades()],
                        [n.inferred_pos['x']['var']**0.5 for n in phylo_tree.find_clades()], [n.inferred_pos['y']['var']**0.5 for n in phylo_tree.find_clades()] ,
                c='k', alpha=0.1, marker='d', ls=None)
        axs[1].scatter([n.pos['x'] for n in phylo_tree.find_clades()], [n.pos['y'] for n in phylo_tree.find_clades()], 
                c=[n.t for n in phylo_tree.find_clades()], s=30)

#                c=[n.t for n in phylo_tree.find_clades()], s=30, marker='d', ls=None)

        x = np.linspace(*axs[1].get_xlim(),101)
        axs[1].plot(x, target_density(x, 0, t)/N, label='target density')

    plt.figure()
    plt.plot(x, target_density(x, 0, t)/100, label='target density')
