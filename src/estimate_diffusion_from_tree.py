import numpy as np

def next_nontrivial_child(node):
    b = node
    while len([c for c in b['children'] if c['alive']])==1:
        i = np.argmax([c['alive'] for c in b['children']])
        b = b['children'][i]
    return b

def clean_tree(tree):
    def recursively_bridge(node):
        node["clades"] = [next_nontrivial_child(c) for c in node['children'] if c['alive']]
        for c in node['clades']:
            recursively_bridge(c)

    recursively_bridge(tree)


def branch_contribution(b, x0, y0, t):
    return {'dx': b['x']-x0, 'dy':b['y'] - y0, 'dt': b['time']-t}


def estimate_diffusion_rec(node, displacements):
    for child in node['clades']:
        data = branch_contribution(child, node['x'], node['y'], node['time'])
        displacements.append(data)
        estimate_diffusion_rec(child, displacements)

# produce empirical estimates of the diffusion constant from the simulated tree annotated with
# positions and time.
def estimate_diffusion(tree, include_root=False):
    displacements = []

    # handle root-parent-branch contribution separately
    if len(tree['clades'])==1 and include_root:
        #advance to first furcating node
        data = branch_contribution(tree, tree['x'], tree['y'], tree['time'])
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
