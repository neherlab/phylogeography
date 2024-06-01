import numpy as np
from habitat_shifts import seasaw
from parse_and_plot import add_panel_label
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
    parser.add_argument('--D', type=float, default=0.0008)
    parser.add_argument('--interaction-radius', type=float, default=0.1)
    parser.add_argument('--density-reg', type=float, default=0.1)
    parser.add_argument('--subsampling', type=float, default=1.0)
    args = parser.parse_args()

    # pick seed to ensure reproducible figure
    np.random.seed(1234)

    N = args.N
    width = 0.5
    Lx, Ly = 3, 1
    D = args.D
    interaction_radius, density_reg = args.interaction_radius, args.density_reg
    period = 500
    target_density = seasaw(N, Lx, Ly, **{'period':period, 'amplitude':1.1})

    v_FKPP = 2*np.sqrt(args.velocity*D)
    print(f"{interaction_radius=:1.3f}, {density_reg=:1.3f}")

    tree = make_node(Lx/2,Ly/2,-2, None)
    tree['children'] = [make_node(np.random.random()*Lx, np.random.random()*Ly, -1, tree)
                        for i in range(N)]
    terminal_nodes = tree['children']

    # generate snapshots and various times and plot tree and population distribution
    t = 0
    for tmax in [5*N + i*period/8.0 for i in range(9)]:
        while t<tmax:
            terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly, interaction_radius=interaction_radius,
                                density_reg=density_reg, D=D, target_density=target_density, total_population=N, periodic=False)
            t+=1

        subsample_tree(terminal_nodes, tree, p=1.0, subtree_attr='clades')
        D_res = estimate_diffusion(tree)
        estimate_ancestral_positions(tree, D_res['Dxy_total'])
        phylo_tree = dict_to_phylo_tree(tree['clades'][0] if len(tree['clades'])==1 else tree,
                                        child_attr='clades')

        fig, axs = plt.subplots(1,2, figsize=(12,3))
        Phylo.draw(phylo_tree, axes=axs[0], label_func=lambda x: '')
        print(f"N={len(terminal_nodes)}")

        # plot true and inferred positions of nodes
        axs[1].scatter([n.pos['x'] for n in phylo_tree.get_nonterminals()],
                       [n.pos['y'] for n in phylo_tree.get_nonterminals()],
                        c=[n.t for n in phylo_tree.get_nonterminals()], s=30)

        axs[1].scatter([n.inferred_pos['x']['mean'] for n in phylo_tree.get_nonterminals()],
                       [n.inferred_pos['y']['mean'] for n in phylo_tree.get_nonterminals()],
                        c=[n.t for n in phylo_tree.get_nonterminals()], s=20, marker='^')

        # highlight true and inferred root positions
        n = phylo_tree.root
        axs[1].scatter([n.pos['x']], [n.pos['y']], c='r', s=100)
        axs[1].scatter([n.inferred_pos['x']['mean']], [n.inferred_pos['y']['mean']], c='g', s=100, marker='^')
#                c=[n.t for n in phylo_tree.find_clades()], s=30, marker='d', ls=None)
        # z = [(n.inferred_pos['x']['mean'] - n.pos['x'])/n.inferred_pos['x']['var']**0.5 for n in phylo_tree.get_nonterminals()]
        # tps = [n.t for n in phylo_tree.get_nonterminals()]
        # axs[2].scatter(tps, z, c='k', s=30)

        # add arrows indicating the link between true and inferred positions
        for n in phylo_tree.get_nonterminals():
            axs[1].arrow(n.pos['x'], n.pos['y'], n.inferred_pos['x']['mean'] - n.pos['x'],
                         n.inferred_pos['y']['mean'] - n.pos['y'], lw=0.5, alpha=0.5)

        # indicate density
        x = np.linspace(0, Lx,101)
        dens = target_density(x, 0, t)
        axs[1].plot(x, dens/np.max(dens), label='target density')
        axs[1].set_xlim(0,Lx)

        plt.tight_layout()

