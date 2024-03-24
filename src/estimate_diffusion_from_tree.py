import numpy as np

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


def estimate_ancestral_positions(tree, D):
    def preorder(node):

        node['dis_to_parent'] = {'x':{'a':0, 'b':0, 'c':0}, 'y':{'a':0, 'b':0, 'c':0}}
        for c in node['clades']:
            dc = 1.0/(4*D*(c['time']-node['time']))
            if 'clades' in c and len(c['clades']):
                preorder(c)
                for x in ['x', 'y']:
                    child_d = c['dis_to_parent'][x]
                    node['dis_to_parent'][x]['a'] += dc*child_d['a']/(dc + child_d['a'])
                    node['dis_to_parent'][x]['b'] += dc*child_d['b']/(dc + child_d['a'])
                    node['dis_to_parent'][x]['c'] += child_d['c'] \
                                        + child_d['b']**2/(dc + child_d['a']) \
                                        + 0.5*np.log(dc/(dc + child_d['a']))
            else:
                for x in ['x', 'y']:
                    node['dis_to_parent'][x]['a'] += dc
                    node['dis_to_parent'][x]['b'] += dc*c[x]
                    node['dis_to_parent'][x]['c'] += dc*c[x]**2 - 0.5*np.log(np.pi/dc)
            for x in ['x', 'y']:
                node['dis_to_parent'][x]['var'] = 2.0/node['dis_to_parent'][x]['a']
                node['dis_to_parent'][x]['mean'] = 0.5*node['dis_to_parent'][x]['b']*node['dis_to_parent'][x]['var']


    preorder(tree)
    tree['dis_from_parent'] = {'x':{'a':0, 'b':0, 'c':0}, 'y':{'a':0, 'b':0, 'c':0}, 'dt': np.inf}
    tree['position'] = {}
    for x in ['x', 'y']:
        d = tree['dis_to_parent'][x]
        tree['position'][x] = {'var': 2.0/d['a'], 'mean': d['b']/d['a']}

    def postorder(node):
        dn = 1.0/(4*D*node['dis_from_parent']['dt'])
        for c1 in node['clades']:
            c1['dis_from_parent'] = {'x':{'a':0, 'b':0, 'c':0}, 'y':{'a':0, 'b':0, 'c':0}, 'dt':c1['time']-node['time']}
            dc = 1.0/(4*D*(c1['time']-node['time']))
            for x in ['x', 'y']:
                if 'clades' in c1 and len(c1['clades']):
                    c1_d = c1['dis_to_parent'][x]
                    c1['dis_from_parent'][x]['a'] = node['dis_to_parent'][x]['a'] - dc*c1_d['a']/(dc + c1_d['a'])
                    c1['dis_from_parent'][x]['b'] = node['dis_to_parent'][x]['b'] - dc*c1_d['b']/(dc + c1_d['a'])
                    c1['dis_from_parent'][x]['c'] = node['dis_to_parent'][x]['c'] - c1_d['c'] - c1_d['b']**2/(dc + c1_d['a']) - 0.5*np.log(dc/(dc + c1_d['a']))
                else:
                    c1['dis_from_parent'][x]['a'] = node['dis_to_parent'][x]['a'] - dc
                    c1['dis_from_parent'][x]['b'] = node['dis_to_parent'][x]['b'] - dc*c1[x]
                    c1['dis_from_parent'][x]['c'] = node['dis_to_parent'][x]['c'] - dc*c1[x]**2 - 0.5*np.log(np.pi/dc)
                if node!=tree:
                    d = node['dis_from_parent'][x]
                    c1['dis_from_parent'][x]['a'] += d['a']*dn/(dn + d['a'])
                    c1['dis_from_parent'][x]['b'] += d['b']*dn/(dn + d['a'])
                    c1['dis_from_parent'][x]['c'] += d['c'] + d['b']**2/(dn + d['a']) + 0.5*np.log(dn/(dn + d['a']))
            postorder(c1)

    postorder(tree)

    def marginal_positions(node):
        node['position'] = {}
        dn = 1/(4*D*node['dis_from_parent']['dt'])
        for x in ['x', 'y']:
            d = node['dis_from_parent'][x]
            if "clades" in node and len(node['clades']):
                n = node['dis_to_parent'][x]
                node['position'][x] = {'var': 0.5/(n['a'] + d['a']*dn/(d['a'] + dn))}
                node['position'][x]['mean'] =  2.0*(n['b'] + d['b']*dn/(d['a'] + dn))*node['position'][x]['var']
            else:
                if (d['a']*dn/(d['a'] + dn))==0:
                    import ipdb; ipdb.set_trace()
                node['position'][x] = {'var': 0.5/(d['a']*dn/(d['a'] + dn))}
                node['position'][x]['mean'] =  2.0*(d['b']*dn/(d['a'] + dn))*node['position'][x]['var']

    def assign_positions(node):
        if node!=tree:
            marginal_positions(node)
        for c in node['clades']:
            assign_positions(c)

    assign_positions(tree)


def collect_positioning(tree):
    res = []
    def collect_positioning_rec(node, res):
        nonterminal = 'clades' in node and len(node['clades'])>0
        res.append({'time': node['time'], 'x': node['x'], 'x_mean': node['position']['x']['mean'], 'x_std': node['position']['x']['var']**0.5,
                    'y': node['y'], 'y_mean': node['position']['y']['mean'], 'y_std': node['position']['y']['var']**0.5, 'nonterminal': nonterminal})
        if nonterminal:
            res[-1]['subtree_x_mean'] = node['dis_to_parent']['x']['mean']
            res[-1]['subtree_x_std'] = node['dis_to_parent']['x']['var']**0.5
            res[-1]['subtree_y_mean'] = node['dis_to_parent']['y']['mean']
            res[-1]['subtree_y_std'] = node['dis_to_parent']['y']['var']**0.5
            for c in node['clades']:
                collect_positioning_rec(c, res)
    collect_positioning_rec(tree, res)
    return res

def collect_zscore(tree):
    import pandas as pd
    res = []
    def collect_zscore_rec(node):
        nonterminal = 'clades' in node and len(node['clades'])>0
        res.append({'zx': (node['x']-node['position']['x']['mean'])/node['position']['x']['var']**0.5,
                    'zy': (node['y']-node['position']['y']['mean'])/node['position']['y']['var']**0.5,
                    't': node['time'], 'nonterminal':nonterminal})
        if nonterminal:
            for c in node['clades']:
                collect_zscore_rec(c)
    collect_zscore_rec(tree)
    return pd.DataFrame(res)
