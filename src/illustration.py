import numpy as np
from free_diffusion import *
from density_regulation import dict_to_phylo_tree
import matplotlib.pyplot as plt
from Bio import Phylo
from matplotlib.cm import viridis_r

if __name__=="__main__":
    n_leaves = 100
    np.random.seed(1234)

    fig, axs = plt.subplots(2,2, figsize=(6,8))
    for row, yule in [(0,True), (1,False)]:
        scale = np.log(n_leaves) if yule else 2
        tree = random_tree(n_leaves, yule=yule)
        add_positions(tree['tree'], 1)
        phylo_tree = dict_to_phylo_tree(tree['tree'], child_attr='clades')
        phylo_tree.ladderize()
        tip_count = 0
        for n in phylo_tree.find_clades(order='postorder'):
            if n.is_terminal():
                n.xvalue=tip_count
                tip_count+=1
            else:
                minx = min([c.xvalue for c in n])
                maxx = max([c.xvalue for c in n])
                n.xvalue = (minx+maxx)/2

            n.color = [int(x) for x in viridis_r(-n.t/scale, bytes=True)[:3]]
        phylo_tree.root.div = phylo_tree.root.branch_length
        for n in phylo_tree.get_nonterminals(order='preorder'):
            for c in n:
                c.div = n.div + c.branch_length


        Phylo.draw(phylo_tree, axes=axs[row,0], label_func=lambda x: '')
        axs[row,0].set_title(f"{'Yule' if yule else 'Kingman'} tree")
        axs[row,0].axis('off')
        for n in phylo_tree.get_nonterminals():
            for c in n.clades:
                axs[row,1].plot([n.pos['x'], c.pos['x']], [n.pos['y'], c.pos['y']], c='k', lw=0.5, alpha=0.5)

        axs[row,1].scatter([n.pos['x'] for n in phylo_tree.find_clades()], [n.pos['y'] for n in phylo_tree.find_clades()],
                c=[n.t for n in phylo_tree.find_clades()], s=30)
        axs[row,1].axis('equal')
        axs[row,1].scatter([0], [0], c='r', s=100)
        axs[row,1].set_axis_off()


        axs[row,0].scatter([n.div for n in phylo_tree.find_clades()], [n.xvalue for n in phylo_tree.find_clades()],
                c=[n.t for n in phylo_tree.find_clades()], s=30)
        axs[row,0].scatter([phylo_tree.root.div], [phylo_tree.root.xvalue], c='r', s=100)

    plt.tight_layout()
    plt.savefig('figures/illustration_tree.pdf')
