import numpy as np

def random_tree(n, yule=False):
    '''
    Generate a random tree either with kingman or yule branching properties
    '''
    def node(t, children=None, node_type="internal", name=None):
        return {"time": t, "children": children or [], "type":node_type, "name":name}

    t = 0
    tree = [node(t, node_type="terminal", name=i) for i in range(n)]
    terminals = [x for x in tree]
    node_count = len(tree)
    while len(tree)>=2:
        i,j = np.random.choice(len(tree), size=2, replace=False)
        if yule:
            dt = np.random.exponential()/(len(tree)-1)/n*2
        else:
            dt = np.random.exponential()/(len(tree)-1)/len(tree)*2

        t -= dt
        new_node = node(t, children=[tree[i], tree[j]], node_type="internal", name=node_count)
        tree.append(new_node)
        node_count += 1
        tree.pop(max(i,j))
        tree.pop(min(i,j))

    return {"tree": tree[0], "terminals": terminals}

def add_positions(tree, D):
    '''
    recursively add positions to the nodes by adding diffusive increments
    '''
    def add_positions_rec(node, D):
        if node["type"] == "internal":
            for child in node["children"]:
                dt = child["time"] - node["time"]
                dx = np.random.randn()*np.sqrt(2*D*dt)
                dy = np.random.randn()*np.sqrt(2*D*dt)
                child['dt'] = dt
                child['dx'] = dx
                child['dy'] = dy
                child['x'] = node['x'] + dx
                child['y'] = node['y'] + dy
                add_positions_rec(child, D)

    tree['x'] = 0
    tree['y'] = 0
    add_positions_rec(tree, D)

