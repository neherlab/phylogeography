import numpy as np
from density_regulation import make_node, evolve, clean_tree, dict_to_phylo_tree, add_as_color, subsample_tree
from heterogeneity import get_2d_hist
from estimate_diffusion_from_tree import estimate_diffusion, estimate_ancestral_positions, collect_zscore
from Bio import Phylo

N=1000
tmax = 10000
density_reg = 0.1
interaction_radius = 0.1
Lx=1
Ly=1
D=0.1*Lx*Ly/N

tree = make_node(Lx/2,Ly/2,-2, None)
tree['children'] = [make_node(np.random.random()*Lx, np.random.random()*Ly, -1, tree)
                    for i in range(N)]
terminal_nodes = tree['children']


for t in range(tmax):
    terminal_nodes = evolve(terminal_nodes, t, Lx=Lx, Ly=Ly, interaction_radius=interaction_radius,
                            density_reg=density_reg, D=D, target_density=N)

clean_tree(tree)
subsample_tree(terminal_nodes, tree, p=1.0)
D_res = estimate_diffusion(tree)
estimate_ancestral_positions(tree, D)
z = collect_zscore(tree)

import matplotlib.pyplot as plt
plt.ion()

T = dict_to_phylo_tree(tree)
from matplotlib.cm import RdBu, Spectral
from Bio import Phylo
add_as_color(T, quantity='x', cmap=RdBu)
Phylo.draw(T)

add_as_color(T, quantity='devx', cmap=Spectral)
def label_func(n):
    return f"{n.pos['x']:1.2f}_{n.inferred_pos['x']['mean']:1.2f}_{n.inferred_pos['x']['var']**0.5:1.2f}"

Phylo.draw(T, label_func=label_func)

