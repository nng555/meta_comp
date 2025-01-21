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
def train(
    data_path: str='kl_diffs/vae_final_vs_vae_final',
    bsize: int=64,
    nepochs: int=10,
    lr: float=1e-3,
):

    model = MetaMLP()
    model.cuda()

    trainset = MetaDataset([data_path + '/train'])
    testset = MetaDataset([data_path + '/test'])

    trainloader = DataLoader(trainset, batch_size=bsize, shuffle=True)
    testloader = DataLoader(testset, batch_size=bsize, shuffle=False)

    batch_per_epoch = len(trainloader)
    optim = Adam(model.parameters(), lr=lr)
    scheduler = SequentialLR(
        optim,
        schedulers=[
            LinearLR(optim, start_factor=1e-3, end_factor=1, total_iters=batch_per_epoch),
            ConstantLR(optim, factor=1, total_iters=(nepochs - 2) * batch_per_epoch),
            LinearLR(optim, start_factor=1, end_factor=1e-3, total_iters=batch_per_epoch),
        ],
        milestones=[batch_per_epoch, (nepochs - 1) * batch_per_epoch],
    )
    optim.zero_grad()

    for epoch in range(nepochs):

        model.train()

        train_loss = 0
        train_acc = 0
        for i, (imgs, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
            out = model(imgs).squeeze()
            loss = ((out - labels)**2).mean()
            #loss = F.binary_cross_entropy_with_logits(out, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()
            train_loss += loss.item()
            train_acc += ((out > 0) == (labels > 0)).float().mean().item()

        train_loss /= len(trainloader)
        train_acc /= len(trainloader)

        print(f"Train Loss: {train_loss:.4f}\tAccuracy: {train_acc:.4f}")

        model.eval()

        test_loss = 0
        test_acc = 0
        for i, (imgs, labels) in tqdm(enumerate(testloader), total=len(testloader)):
            out = model(imgs).squeeze()
            loss = ((out - labels)**2).mean()
            #loss = F.binary_cross_entropy_with_logits(out, labels)
            test_loss += loss.item()
            test_acc += ((out > 0) == (labels > 0)).float().mean().item()

        test_loss /= len(testloader)
        test_acc /= len(testloader)

        print(f"Test Loss: {test_loss:.4f}\tAccuracy: {test_acc:.4f}")

    torch.save(model, '/scratch/nhn234/checkpoints/meta_model.pt')

if __name__ == "__main__":
    app()
