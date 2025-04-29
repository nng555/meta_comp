import sys
sys.path.append("/scratch/mr7401/projects/meta_comp")
from datasets import load_dataset
from collections import defaultdict
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from model import VAE
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
import wandb
import os
os.environ["HF_HOME"] = "/scratch/" + os.environ["USER"] + "/cache"

import typer
import pandas as pd
app = typer.Typer()

IMAGE_SIZE=28

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x)),
])

def collator(batch):
    batch = {
        'image': torch.stack([b['image'] for b in batch])
    }
    return batch

@app.command()
def train(
    nepochs: int=50,
    latent_dim: int=128,
    bsize: int=64,
    lr: float=1e-3,
    kl_weight=1.0,
):
    kl_weight = float(kl_weight)

    run = wandb.init(
        project="nts_mnist",
        name=f"vae_ldim_{latent_dim}_klw_{kl_weight}",
        config={
            "nepochs": nepochs,
            "latent_dim": latent_dim,
            "bsize": bsize,
            "lr": lr,
            'kl_weight': kl_weight,
        }
    )


    # Check for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load dataset
    data = load_dataset('ylecun/mnist', split='train')
    data = data.train_test_split(test_size=0.01)

    # set transforms
    def mnist_t(examples):
        examples['image'] = [mnist_transform(image) for image in examples['image']]
        return examples
    data.set_transform(mnist_t) # 784 flat

    # load model
    model = VAE(
        in_dim=784,
        latent_dim=latent_dim, # maybe select them some other way?
        device=device,
    )
    model.to(device)

    batch_per_epoch = int(len(data['train']) / bsize)
    test_batch_per_epoch = int(len(data['test']) / bsize)


    loader = DataLoader(data['train'], batch_size=bsize, collate_fn=collator, shuffle=True)
    test_loader = DataLoader(data['test'], batch_size=bsize, collate_fn=collator, shuffle=True)

    optim = Adam(model.parameters(), lr=lr)
    scheduler = SequentialLR(
        optim,
        schedulers=[
            LinearLR(optim, start_factor=1e-3, end_factor=1, total_iters=3 * batch_per_epoch),
            ConstantLR(optim, factor=1, total_iters=(nepochs - 6) * batch_per_epoch),
            LinearLR(optim, start_factor=1, end_factor=1e-3, total_iters=3 * batch_per_epoch),
        ],
        milestones=[batch_per_epoch, (nepochs - 1) * batch_per_epoch],
    )
    optim.zero_grad()

    loss_averages = []
    # train
    for epoch in range(nepochs):
        epoch_loss = defaultdict(float)

        for i, batch in tqdm(enumerate(loader), total=batch_per_epoch):
            out = model(batch['image'].to(device))
            out[-1] = torch.clamp_(out[-1], -10, 10)
            loss = model.loss_function(*out, M_N=kl_weight)
            loss['loss'].backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

            for k, l in loss.items():
                epoch_loss[k] += l.item() * len(batch['image'])

            epoch_loss['mu'] += torch.abs(out[-2]).mean()
            epoch_loss['log_var'] += out[-1].mean()

        # TODO: don't hardcode this
        if (epoch + 1) % 10 == 0:
            torch.save(model, f'/home/nhn234/projects/meta_comp/mnist/checkpoints/vae_ldim_{latent_dim}_e{epoch + 1}.pt')

        model.eval()
        test_loss = defaultdict(float)
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                out = model(batch['image'].cuda())
                loss = model.loss_function(*out, M_N=kl_weight)
                for k, l in loss.items():
                    test_loss[k] += l.item() * len(batch['image'])
                test_loss['pixel_acc'] += (torch.round(out[0]) == torch.round(out[1])).float().mean().item() * len(batch['image'])

        logs = {}
        for k, v in epoch_loss.items():
            logs['train/' + k] = v / len(data['train'])
        for k, v in test_loss.items():
            logs['test/' + k] = v / len(data['test'])

        logs['test_loss'] = test_loss
        logs['lr'] = scheduler.get_last_lr()[0]
        wandb.log(logs)

if __name__ == "__main__":
    app()
