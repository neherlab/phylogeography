import numpy as np



def random_tree(n, yule=False):
    def node(t, children=None, node_type="internal", name=None):
        return {"time": t, "children": children or [], "type":node_type, "name":name}

    t = 0
    tree = [node(t, node_type="terminal", name=i) for i in range(n)]
    node_count = len(tree)
    while len(tree)>=2:
        i,j = np.random.choice(len(tree), size=2, replace=False)
        if yule:
            dt = np.random.exponential()/(len(tree)-1)
        else:
            dt = np.random.exponential()/(len(tree)-1)/len(tree)*2

        t += dt
        new_node = node(t, children=[tree[i], tree[j]], node_type="internal", name=node_count)
        tree.append(new_node)
        node_count += 1
        tree.pop(max(i,j))
        tree.pop(min(i,j))

    return tree[0]

def add_positions(tree, D):
    def add_positions_rec(node, D):
        if node["type"] == "internal":
            for child in node["children"]:
                dt = node["time"] - child["time"]
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

def empirical_diffusion(tree):
    def empirical_diffusion_rec(node):
        if node["type"] == "internal":
            for child in node["children"]:
                data.append((child['dt'], child['dx'], child['dy']))
                empirical_diffusion_rec(child)

    data = []
    empirical_diffusion_rec(tree)
    data = np.array(data)
    return {"data": data, "Dx": 0.5*np.mean(data[:,1]**2/data[:,0]), "Dy": 0.5*np.mean(data[:,2]**2/data[:,0]),
            "bDx": 0.5*np.mean(data[:,1]**2)/np.mean(data[:,0]), "bDy": 0.5*np.mean(data[:,2]**2)/np.mean(data[:,0]),
            "vx": np.mean(np.abs(data[:,1])/data[:,0]), "vy": np.mean(np.abs(data[:,2])/data[:,0]),
            "bvx": np.mean(np.abs(data[:,1]))/np.mean(data[:,0]), "bvy": np.mean(np.abs(data[:,2]))/np.mean(data[:,0])}


if __name__ == "__main__":

    yule = True
    replicates = 1000
    results = {}
    for n in [10,30, 100, 300, 1000]:
        res = []
        for i in range(replicates):
            tree = random_tree(n, yule=yule)
            add_positions(tree, 1)
            D = empirical_diffusion(tree)
            res.append((D['data'][:,0].mean(), D["Dx"], D["Dy"], D["vx"], D["vy"], D["bDx"], D["bDy"], D["bvx"], D["bvy"]))

        res = np.array(res)
        print(f"{n=}")
        results[n] = {"branch_length": res[:,0].mean(),
                      "Dx": res[:,1].mean(), "Dy": res[:,2].mean(), "vx": res[:,3].mean(), "vy": res[:,4].mean(),
                      "bDx": res[:,5].mean(), "bDy": res[:,6].mean(), "bvx": res[:,7].mean(), "bvy": res[:,8].mean(),
                      "bDx_std": res[:,5].std(), "bDy_std": res[:,6].std(), "bvx_std": res[:,7].std(), "bvy_std": res[:,8].std(),
                      "Dx_std": res[:,1].std(), "Dy_std": res[:,2].std(), "vx_std": res[:,3].std(), "vy_std": res[:,4].std()}


    import matplotlib.pyplot as plt

    for prefix in ['', 'b']:
        plt.figure()
        for p in ['Dx', 'vx']:
            q = prefix+p
            label = f"{q} [{'mean(dx_i^2)/mean(2t_i)' if 'D' in q else 'mean(dx_i)/mean(t_i)'}]" if prefix else f"{q} [{'mean(dx_i^2/2t_i)' if 'D' in q else 'mean(dx_i/t_i)'}]"
            plt.errorbar([n for n in results], [results[n][q] for n in results], [results[n][q+"_std"] for n in results], ls = '-', marker='o',
                        label=label)

        plt.xlabel('Number of leaves')
        plt.ylabel('estimate')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(f'{"yule_" if yule else "kingman_"}{"branch_" if prefix else ""}dispersal.png')
