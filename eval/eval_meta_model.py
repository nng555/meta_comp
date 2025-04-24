from meta_model import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

import typer
app = typer.Typer()

@app.command()
def eval(
    model_path: str='/scratch/nhn234/checkpoints/meta_model.pt',
    bsize: int=64,
):

    model = torch.load(model_path)
    data_path ='kl_diffs/vae_final_vs_vae_final'
    fullset = MetaDataset([data_path + '/train', data_path + '/test'])

    accs = defaultdict(float)
    losses = defaultdict(float)

    model.eval()

    for name, data, labels in zip(fullset.names, fullset.data, fullset.labels):
        subset = MetaSubDataset(data, labels, name)
        subloader = DataLoader(subset, batch_size=bsize)

        for i, (imgs, labels) in tqdm(enumerate(subloader), total=len(subloader)):
            out = model(imgs).squeeze()
            loss = ((out - labels)**2).sum()
            #loss = F.binary_cross_entropy_with_logits(out, labels)
            losses[name] += loss.item()
            accs[name] += ((out > 0) == (labels > 0)).float().sum().item()

        losses[name] /= len(subset)
        accs[name] /= len(subset)


    name_list = [
        'ldim_8',
        'ldim_16',
        'ldim_32',
        'ldim_64',
        'ldim_128',
        'ldim_192',
        'ldim_256',
        'ldim_384',
        'ldim_512',
    ]

    acc_data = np.zeros((len(name_list), len(name_list)))
    loss_data = np.zeros((len(name_list), len(name_list)))
    for i, n1 in enumerate(name_list):
        for j, n2 in enumerate(name_list):
            acc_data[i][j] = accs[n1 + '_vs_' + n2]
            loss_data[i][j] = losses[n1 + '_vs_' + n2]

    sns.heatmap(acc_data, annot=True, xticklabels=name_list, yticklabels=name_list, vmin=0, vmax=1)
    plt.savefig("/home/nhn234/acc.png")

    plt.clf()
    plt.cla()
    sns.heatmap(loss_data, annot=True, xticklabels=name_list, yticklabels=name_list)
    plt.savefig("/home/nhn234/loss.png")

if __name__ == "__main__":
    app()
