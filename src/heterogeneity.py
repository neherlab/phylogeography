from free_diffusion import *
import matplotlib.pyplot as plt

def get_2d_hist(terminal_nodes, Lx, Ly, bins):
    x_pos = (np.array([c['x'] for c in terminal_nodes])+Lx/2)%Lx
    y_pos = (np.array([c['y'] for c in terminal_nodes])+Ly/2)%Ly
    return np.histogram2d(x_pos, y_pos, bins=(np.linspace(0,Lx,bins+1),np.linspace(0,Ly,bins+1)))

def get_granularity(ntips, D, yule=False, L=1, bins=10):
    T = random_tree(ntips, yule=yule)
    add_positions(T['tree'], D)
    H, bx, by = get_2d_hist(T['terminals'], L, L, bins)
    return np.std(H)/ntips*np.prod(H.shape)


if __name__=="__main__":
    n_iter = 10
    ntips = 1000
    linear_bins=5
    res = []
    D_array = np.logspace(-2,2,101)
    for D in D_array:
        res.append([get_granularity(ntips, D, bins=linear_bins) for i in range(n_iter)])

    res = np.array(res)
    res_mean = res.mean(axis=1)

    nbins=linear_bins**2
    area = np.minimum(1,D_array*np.pi**2)
    pred_heterogeneity = np.sqrt(1/area - 1)  # p*(1/p - 1)**2 + (1-p) = p*(1/p^2 - 2/p + 1) + (1-p) = 1/p - 2 + p -1 + p = 1/p - 1

    plt.plot(D_array, res_mean, label='free diffusion')
    plt.plot(D_array, np.ones_like(D_array)*np.sqrt(nbins/ntips), label='well mixed limit')
    #plt.plot(D_array, pred_heterogeneity, label='well mixed limit')
    plt.xscale('log')
    plt.xlabel('diffusion constant')
    plt.ylabel('heterogeneity')
    plt.savefig('figures/heterogeneity_free_diffusion.png')
