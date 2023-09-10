from free_diffusion import *
import matplotlib.pyplot as plt

def get_2d_hist(terminal_nodes, Lx, Ly, bins):
    x_pos = (np.array([c['x'] for c in terminal_nodes]))%Lx
    y_pos = (np.array([c['y'] for c in terminal_nodes]))%Ly
    assert Ly<=Lx
    aspect=Lx/Ly
    return np.histogram2d(x_pos, y_pos, bins=(np.linspace(0,Lx,int(aspect*bins+1)),np.linspace(0,Ly,bins+1)))

def get_granularity(ntips, D, yule=False, L=1, bins=10):
    T = random_tree(ntips, yule=yule)
    add_positions(T['tree'], D)
    H, bx, by = get_2d_hist(T['terminals'], L, L, bins)
    return np.std(H)/ntips*np.prod(H.shape), H


if __name__=="__main__":
    n_iter = 10
    ntips = 1000
    D_array = np.logspace(-2,2,101)

    fig, axs = plt.subplots(1,2, figsize=(12,6))

    g, H = get_granularity(ntips, 0.1, bins=20)
    axs[0].matshow(H)

    for line_style, linear_bins in zip(['-', '--', '-.'], [3,5,10]):
        nbins=linear_bins**2
        res = []
        for D in D_array:
            res.append([get_granularity(ntips, D, bins=linear_bins)[0] for i in range(n_iter)])

        res = np.array(res)
        res_mean = res.mean(axis=1)
        axs[1].plot(D_array, res_mean, label=f'free diffusion, dx=L/{linear_bins}', c='C0', ls=line_style)
        axs[1].plot(D_array, np.ones_like(D_array)*np.sqrt(nbins/ntips), label=f'well mixed limit dx=L/{linear_bins}', c='C1', ls=line_style)

    # area = np.minimum(1,D_array*np.pi**2)
    # pred_heterogeneity = np.sqrt(1/area - 1)  # p*(1/p - 1)**2 + (1-p) = p*(1/p^2 - 2/p + 1) + (1-p) = 1/p - 2 + p -1 + p = 1/p - 1

    #plt.plot(D_array, pred_heterogeneity, label='well mixed limit')
    plt.xscale('log')
    plt.xlabel('diffusion constant [habitat_size^2/T_c]')
    plt.ylabel('heterogeneity')
    plt.legend()
    plt.savefig('figures/heterogeneity_free_diffusion.pdf')
    plt.savefig('figures/heterogeneity_free_diffusion.pdf')
