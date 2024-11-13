import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch

def plot_kl_diffs(data_name):
    ncomps = 7
    names = [
        "ldim_2",
        "ldim_4",
        "ldim_8",
        "ldim_16",
        "ldim_32",
        "ldim_64",
        "ldim_128",
    ]

    #ncomps = 15
    #names = [f'e{i}' for i in range(15)]

    fig, axs = plt.subplots(
        ncols=ncomps, nrows=ncomps,
        figsize=(45, 30)
    )
    #plt.subplots_adjust(left=0.04, top=0.07)

    for col in range(ncomps):
        ax = axs[0, col]
        ax.set_title(names[col], pad=20, fontsize=30)

    for row in range(ncomps):
        ax = axs[row, 0]
        ax.set_ylabel(names[row], labelpad=20, fontsize=30)

    data_path = '/scratch/nhn234/data/kl_diffs/' + data_name

    for row in range(ncomps):
        for col in range(ncomps):
            plot_path = os.path.join(
                data_path,
                names[row] + '_vs_' + names[col] + '.pt_diff',
            )
            data = torch.load(plot_path, map_location=torch.device('cpu')).numpy()
            #data.reshape(10, 1000).mean(0)
            sns.kdeplot(data, ax=axs[row][col])
            axs[row][col].axvline(x=0, color='black', linestyle=':', alpha=0.5)

    fig.savefig("/home/nhn234/kl_diff.png")

if __name__ == "__main__":
    plot_kl_diffs('vae_final_vs_vae_final')
    #plot_kl_diffs('vae_ldim_128_vs_vae_ldim_128')
