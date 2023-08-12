import numpy as np

def branch_contribution(b, x0, y0, t, non_trivial_nodes):
    while len([c for c in b['children'] if c['alive']])==1:
        i = np.argmax([c['alive'] for c in b['children']])
        b = b['children'][i]
    return {'dx': b['x']-x0, 'dy':b['y'] - y0, 'dt': b['time']-t, 'next_node':b}

def estimate_diffusion_rec(node, displacements, non_trivial_nodes=False):
    for child in node['children']:
        data = branch_contribution(child, node['x'], node['y'], node['time'], non_trivial_nodes)
        displacements.append(data)
        estimate_diffusion_rec(data['next_node'], displacements, non_trivial_nodes)

# produce empirical estimates of the diffusion constant from the simulated tree annotated with
# positions and time.
def estimate_diffusion(tree, include_root=False, non_trivial_nodes=False):
    displacements = []

    # handle root-parent-branch contribution separately
    if len(tree['children'])==1:
        #advance to first furcating node
        data = branch_contribution(tree, tree['x'], tree['y'], tree['time'], non_trivial_nodes)
        c = data['next_node']
        if include_root: # otherwise ignore
            displacements.append(data)
    else:
        c=tree

    estimate_diffusion_rec(c, displacements)

    return {'Dx_branch': 0.5*np.mean([d['dx']**2/d['dt'] for d in displacements]),
            'Dy_branch': 0.5*np.mean([d['dy']**2/d['dt'] for d in displacements]),
            'vx_branch': 0.5*np.mean([np.abs(d['dx'])/d['dt'] for d in displacements]),
            'vy_branch': 0.5*np.mean([np.abs(d['dy'])/d['dt'] for d in displacements]),
            'Dx_total': 0.5*np.sum([d['dx']**2 for d in displacements])/np.sum([d['dt'] for d in displacements]),
            'Dy_total': 0.5*np.sum([d['dy']**2 for d in displacements])/np.sum([d['dt'] for d in displacements]),
            'vx_total': 0.5*np.sum(np.abs([d['dx'] for d in displacements])/np.sum([d['dt'] for d in displacements])),
            'vy_total': 0.5*np.sum(np.abs([d['dy'] for d in displacements])/np.sum([d['dt'] for d in displacements]))
            }
