from free_diffusion import *
import matplotlib.pyplot as plt
from parse_and_plot import *

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

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    # fig, axs = plt.subplots(2,1, figsize=(5,9), gridspec_kw={'height_ratios': [2, 1]})
    # ax = axs[0]
    g, H = get_granularity(ntips, 0.1, bins=20)
    ax.matshow(H)
    ax.set_axis_off()
    add_panel_label(ax, 'A')

    # for line_style, linear_bins in zip(['-', '--', '-.'], [3,10,30]):
    #     nbins=linear_bins**2
    #     res = []
    #     for D in D_array:
    #         res.append([get_granularity(ntips, D, bins=linear_bins)[0] for i in range(n_iter)])

    #     res = np.array(res)
    #     res_mean = res.mean(axis=1)
    #     axs[1].plot(D_array, res_mean/np.sqrt(nbins/ntips), label=f'bin size=L/{linear_bins}') #, c='C0', ls=line_style)
    #     # axs[1].plot(D_array, np.ones_like(D_array)*np.sqrt(nbins/ntips), label=f'well mixed limit dx=L/{linear_bins}', c='C1', ls=line_style)

    # # area = np.minimum(1,D_array*np.pi**2)
    # # pred_heterogeneity = np.sqrt(1/area - 1)  # p*(1/p - 1)**2 + (1-p) = p*(1/p^2 - 2/p + 1) + (1-p) = 1/p - 2 + p -1 + p = 1/p - 1

    # #plt.plot(D_array, pred_heterogeneity, label='well mixed limit')
    # add_panel_label(axs[1], 'B')
    # plt.xscale('log')
    # #plt.yscale('log')
    # plt.xlabel(r'diffusion constant $[L^2/T_c]$')
    # plt.ylabel('rel. heterogeneity')
    # plt.legend()
    #plt.tight_layout()
    plt.savefig('figures/heterogeneity_free_diffusion.pdf')
