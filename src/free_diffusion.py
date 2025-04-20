import numpy as np

def random_tree(n, yule=False):
    '''
    Generate a random tree either with kingman or yule branching properties
    '''
    def node(t, clades=None, node_type="internal", name=None):
        return {"time": t, "clades": clades or [], "type":node_type, "name":name, "alive":True }

    t = 0
    tree = [node(t, node_type="terminal", name=i) for i in range(n)]
    terminals = [x for x in tree]
    node_count = len(tree)
    while len(tree)>=2:
        i,j = np.random.choice(len(tree), size=2, replace=False)
        if yule:
            dt = np.random.exponential()/(len(tree)-1)
        else:
            dt = np.random.exponential()/(len(tree)-1)/len(tree)*2

        t -= dt
        new_node = node(t, clades=[tree[i], tree[j]], node_type="internal", name=node_count)
        tree.append(new_node)
        node_count += 1
        tree.pop(max(i,j))
        tree.pop(min(i,j))

    return {"tree": tree[0], "terminals": terminals}

def add_positions(tree, D, gauss=True, exponent=3, generation_time=0.01):
    '''
    recursively add positions to the nodes by adding diffusive increments
    '''
    from scipy.stats import genpareto
    pareto_mean = genpareto.mean(1/(exponent-1))
    def add_positions_rec(node, D):
        if node["type"] == "internal":
            for child in node["clades"]:
                dt = child["time"] - node["time"]
                if gauss:
                    dx = np.random.randn()*np.sqrt(2*D*dt)
                    dy = np.random.randn()*np.sqrt(2*D*dt)
                else:
                    # (1+cx)^(-1 - 1/c) --> e = 1 + 1/c --> c = 1/(e-1)
                    # mean = int_0^inty x (1+x)^-e dx =  int_0^inty (1+x)/(1+x)^e - 1/(1+x)^e dx
                    # = 1/(e-2) - 1/(e-1) = ((e-1) - (e-2))/(e-1)/(e-2) = 1/(e-1)/(e-2)
                    ngen = np.random.poisson(dt/generation_time)
                    dx=0
                    dy=0
                    for i in range(ngen):
                        # generate a pareto distributed step
                        angle = np.random.rand()*2*np.pi
                        d = genpareto.rvs(1/(exponent-1))*np.sqrt(2*D*generation_time)/pareto_mean
                        dx += np.cos(angle)*d
                        dy += np.sin(angle)*d

                child['dt'] = dt
                child['dx'] = dx
                child['dy'] = dy
                child['x'] = node['x'] + dx
                child['y'] = node['y'] + dy
                add_positions_rec(child, D)

    tree['x'] = 0
    tree['y'] = 0
    add_positions_rec(tree, D)

