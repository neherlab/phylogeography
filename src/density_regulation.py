import numpy as np
from heterogeneity import get_2d_hist

def make_node(x,y,t,parent):
    return {'children':[], 'x':x,'y':y, 'n_offspring':0, 'time':t, 'parent':parent, 'alive':True}

def calc_density(terminal_nodes, interaction_radius, Lx, Ly, bins_per_std = 5.0):
    '''
    generate an interpolation object that returns to local population density after
    Gaussian smoothing with an interaction_radius. The function assumes Lx>=Ly.
    '''
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator
    nbins = int(np.ceil(bins_per_std/interaction_radius*Ly))
    H, bx, by = get_2d_hist(terminal_nodes, Lx, Ly, nbins)
    dx = bx[1]-bx[0]

    dens = gaussian_filter(H, interaction_radius/dx, mode='wrap')
    dens *= 1.0/dx**2/dens.sum()*len(terminal_nodes)

    return RegularGridInterpolator(((bx[1:]+bx[:-1])*0.5, (by[1:]+by[:-1])*0.5), dens, method='linear', bounds_error=False, fill_value=None)

def evolve(terminals, t, Lx=1, Ly=1, D=0.1, target_density = 100.0, density_reg = 0.1,
           interaction_radius = 0.1, global_pop_reg=True, total_population=100):
    '''
    step the population forward. The population is a list of terminal nodes
    that generates a number of offspring dependent on local density.
    '''

    # calculate local density at for each extant individual
    density = calc_density(terminals, interaction_radius, Lx, Ly)
    dens_array = density([(c['x']%Lx, c['y']%Ly) for c in terminals])
    if callable(target_density):
        target_density_vals = np.array([target_density(c['x'], c['y'], t) for c in terminals])
    else:
        target_density_vals = target_density

    # calculate fitness of each extant individual
    fitness = np.maximum(0.1,1 + density_reg*(1-dens_array/target_density_vals))
    #print(fitness.mean(), dens_array.mean(), target_density, len(terminals))
    if global_pop_reg: # add global density regulation (set average fitness to one, add density independent term)
        fitness += (1 - fitness.mean()) + 0.1*(1-len(terminals)/total_population)

    # determine offspring number and generate new population
    offspring = np.random.poisson(np.maximum(0.001,fitness))
    new_terminals = []
    diff_std = np.sqrt(2*D)
    for noff, parent in zip(offspring, terminals):
        for c in range(noff):
            dx,dy = np.random.randn(2)*diff_std
            parent['children'].append(make_node(parent['x']+dx, parent['y']+dy, t, parent))
        new_terminals.extend(parent['children'])

    if len(new_terminals)<10:
        print("new nodes:", len(new_terminals))
    # prune branches that didn't yield any offspring. Loop over parent generation "terminals"
    for n in terminals:
        if n['children']: continue
        # walk up the dead branch until a node with offspring is found
        x = n
        while len(x['children'])==0 and x['parent']:
            x['parent']['children'] = [c for c in x['parent']['children'] if c!=x]
            x = x['parent']

    # bridge branches that only have a single child. Loop over parent generation "terminals"
    for n in terminals:
        if len(n['children'])!=1: continue
        tip = n['children'][0]
        x = n
        while len(x['children'])==1 and x['parent']:
            x['parent']['children'] = [c for c in x['parent']['children'] if c!=x] + [x['children'][0]]
            x = x['parent']
            tip['parent'] = x

    return new_terminals

def set_sampled_rec(node, subtree_attr):
    sampled_children = 0
    node[subtree_attr] = []
    if len(node['children']):
        for child in node['children']:
            set_sampled_rec(child, subtree_attr)
            if child['sampled']:
                sampled_children += 1
                node[subtree_attr].append(child)

        node['sampled'] = sampled_children>0

def make_subtree_rec(node, subtree_attr='clades'):
    node[subtree_attr] = [next_nontrivial_child(c, subtree_attr=subtree_attr) for c in node[subtree_attr]]
    for child in node[subtree_attr]:
        make_subtree_rec(child, subtree_attr=subtree_attr)

def subsample_tree(terminal_nodes, tree, p=0.1, subtree_attr='clades'):
    sampled = np.random.random(len(terminal_nodes))<p
    if sampled.sum()<2:
        print(f"not enough samples: {sampled.sum()} out of {len(terminal_nodes)}")

    for state, n in zip(sampled, terminal_nodes):
        n['sampled'] = state

    set_sampled_rec(tree, subtree_attr=subtree_attr)
    make_subtree_rec(tree, subtree_attr=subtree_attr)
    return sampled.sum()

def next_nontrivial_child(node, subtree_attr='children'):
    b = node
    while len(b[subtree_attr])==1:
        b = b[subtree_attr][0]
    return b

def clean_tree(tree):
    def recursively_bridge(node):
        node["children"] = [next_nontrivial_child(c) for c in node['children']]
        for c in node['children']:
            c['parent'] = node
            recursively_bridge(c)

    recursively_bridge(tree)


def dict_to_phylo_tree(d, child_attr='children'):
    '''
    Convert custom tree to a Biopython tree.
    '''
    from Bio.Phylo.BaseTree import Clade, Tree

    def clade_from_dict(d, parent=None):
        clade = Clade()
        clade.name = f'{d["time"]:1.2f}_{d["x"]:1.2f}_{d["y"]:1.2f}'
        clade.branch_length = d['time'] - parent.t if parent else 0.001
        clade.t = d['time']
        clade.pos = {'x': d['x'], 'y': d['y']}
        if 'position' in d:
            clade.inferred_pos = d['position']
        return clade

    def add_clades(clade, d):
        for k in d[child_attr]:
            new_clade = clade_from_dict(k, clade)
            clade.clades.append(new_clade)
            add_clades(new_clade, k)

    tree = Tree()
    tree.root = clade_from_dict(d, None)
    add_clades(tree.root,d)

    return tree

def add_as_color(tree, quantity = 'x', cmap = None):
    if cmap is None:
        from matplotlib.cm import RdBu
        cmap = RdBu

    def value(n):
        if quantity=='x':
            return n.pos['x']
        elif quantity == 'devx':
            return 0.5 + 0.2*(n.pos['x']-n.inferred_pos['x']['mean'])/n.inferred_pos['x']['var']**0.5


    for n in tree.find_clades():
        n.color = [int(255*x) for x in  cmap(value(n))[:3]]

if __name__=="__main__":
    import matplotlib.pyplot as plt
    plt.ion()

    N = 200
    L = 1
    D = 0.1*L**2/N
    tree = make_node(L/2,L/2,-2, None)
    tree['children'] = [make_node(np.random.random()*3*L, np.random.random()*L, -1, tree) for i in range(N)]
    terminal_nodes = tree['children']
    interaction_radius = 0.2
    density = calc_density(terminal_nodes, interaction_radius, 3*L, L)
    density_reg = 0.2
    print(f"starting density: {float(density([0,0]).squeeze()):1.2f}")
    for t in range(3*N):
        terminal_nodes = evolve(terminal_nodes, t, Lx=3*L, Ly=L, interaction_radius=interaction_radius,
                                density_reg=density_reg, D=D, target_density=N/3, 
                                global_pop_reg=False, total_population=N)
        if t%(N/5)==0: print(t, len(terminal_nodes))
    clean_tree(tree)
    T = dict_to_phylo_tree(tree)
    from Bio import Phylo
    Phylo.draw(T)

    subsample_tree(terminal_nodes, tree, p=0.1, subtree_attr='clades')
    T1 = dict_to_phylo_tree(tree, child_attr='clades')
    from Bio import Phylo
    Phylo.draw(T1)

    from heterogeneity import get_2d_hist
    H, bx, by = get_2d_hist(terminal_nodes, 3*L, L, 20)
    plt.figure()
    plt.matshow(H)
