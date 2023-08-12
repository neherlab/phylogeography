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

    return RegularGridInterpolator(((bx[1:]+bx[:-1])*0.5, (by[1:]+by[:-1])*0.5), dens,
                                   method='linear', bounds_error=False, fill_value=None)

def evolve(terminals, t, Lx=1, Ly=1, D=0.1, target_density = 100.0, density_reg = 0.1,
           interaction_radius = 0.1, global_pop_reg=True):
    '''
    step the population forward. The population is a list of terminal nodes
    that generates a number of offspring dependent on local density.
    '''

    # calculate local density at for each extant individual
    density = calc_density(terminals, interaction_radius, Lx, Ly)
    dens_array = density([(c['x']%Lx, c['y']%Ly) for c in terminals])

    # calculate fitness of each extant individual
    fitness = np.maximum(0.1,1 + density_reg*(1-dens_array/target_density))
    #print(fitness.mean(), dens_array.mean(), target_density, len(terminals))
    if global_pop_reg: # add global density regulation (set average fitness to one, add density independent term)
        fitness += (1 - fitness.mean()) + 0.1*(1-len(terminals)/(target_density*Lx*Ly))

    # determine offspring number and generate new population
    offspring = np.random.poisson(np.maximum(0.001,fitness))
    new_terminals = []
    diff_std = np.sqrt(2*D)
    for noff, parent in zip(offspring, terminals):
        for c in range(noff):
            dx,dy = np.random.randn(2)*diff_std
            parent['children'].append(make_node(parent['x']+dx, parent['y']+dy, t, parent))
        new_terminals.extend(parent['children'])

    # prune branches that didn't yield any offspring.
    for n in terminals:
        if n['children']: continue
        x = n
        while len(x['children'])==0:
            x['parent']['children'] = [c for c in x['parent']['children'] if c!=x]
            x = x['parent']

    return new_terminals

def dict_to_phylo_tree(d):
    '''
    Convert custom tree to a Biopython tree. Very slow due to many trivial nodes.
    '''
    from Bio.Phylo.BaseTree import Clade, Tree

    def clade_from_dict(d, parent=None):
        clade = Clade()
        clade.name = 'root'
        clade.branch_length = d['time'] - parent.t if parent else 0.001
        clade.t = d['time']
        clade.pos = [d['x'], d['y']]
        return clade

    def add_clades(clade, d):
        for k in d['children']:
            new_clade = clade_from_dict(k, clade)
            clade.clades.append(new_clade)
            add_clades(new_clade, k)

    tree = Tree()
    tree.root = clade_from_dict(d, None)
    add_clades(tree.root,d)

    return tree


if __name__=="__main__":
    N = 2000
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
                                density_reg=density_reg, D=D, target_density=N/3, global_pop_reg=False)
        if t%(N/5)==0: print(t, len(terminal_nodes))
    T = dict_to_phylo_tree(tree)
    from Bio import Phylo
    Phylo.draw(T)

    from heterogeneity import get_2d_hist
    import matplotlib.pyplot as plt
    H, bx, by = get_2d_hist(terminal_nodes, 3*L, L, 20)
    plt.figure()
    plt.matshow(H)
