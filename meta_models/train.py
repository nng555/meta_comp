import sys
sys.path.append("/scratch/mr7401/projects/meta_comp")
from datasets import load_dataset
from collections import defaultdict
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from models.vae import VAE
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
import os

import typer
import pandas as pd
app = typer.Typer()

IMAGE_SIZE=150

celeb_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()])  # used when transforming image to tensor

def collator(batch):
    batch = {
        'image': torch.stack([b['image'] for b in batch])
    }
    return batch

@app.command()
def train(

    nepochs: int=15,
    latent_dim: int=128,
    bsize: int=16,
    lr: float=1e-3,
):

    # Check for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training configuration: nepochs={nepochs}, latent_dim={latent_dim}, bsize={bsize}, lr={lr}")
    # load dataset

    data = load_dataset('nielsr/CelebA-faces', split="train")
    data = data.train_test_split(test_size=0.01, seed = 42)

    # set transforms
    def celeb_t(examples):
        examples['image'] = [celeb_transform(image) for image in examples['image']]
        return examples
    data.set_transform(celeb_t) # 150 x 150 x 3

    # load model
    model = VAE(
        in_channels=3,
        latent_dim=latent_dim, # maybe select them some other way?
        device=device,
    )
    model.to(device)

    batch_per_epoch = int(len(data['train']) / bsize)

    loader = DataLoader(data['train'], batch_size=bsize, collate_fn=collator, shuffle=True)
    test_loader = DataLoader(data['test'], batch_size=bsize, collate_fn=collator, shuffle=True)

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

    loss_averages = []
    # train
    for epoch in range(nepochs):
        epoch_loss = defaultdict(float)
        batch_loss = defaultdict(float)

        for i, batch in tqdm(enumerate(loader), total=batch_per_epoch):
            out = model(batch['image'].to(device))
            out[-1] = torch.clamp_(out[-1], -10, 10)
            loss = model.loss_function(*out, M_N=0.00025)
            loss['loss'].backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

            for k, l in loss.items():
                epoch_loss[k] += l.item()
                batch_loss[k] += l.item()

            if (i + 1) % 1000 == 0:
                res_str = f"Iter {i+1}/{batch_per_epoch}: ({scheduler.get_last_lr()[0]:.4f})"
                for k, l in batch_loss.items():
                    res_str += f" {k}: {l / 1000:.4f}"
                print(res_str, flush=True)
                batch_loss = defaultdict(float)

     
        print(f"Epoch {epoch + 1}/{nepochs}: loss {epoch_loss['loss'] / batch_per_epoch}")

        print(f"Epoch {epoch + 1}/{nepochs}: loss {epoch_loss['loss'] / batch_per_epoch}")
        # Save epoch loss average to a pandas DataFrame
        if epoch == 0:
            loss_df = pd.DataFrame(columns=['epoch', 'loss_avg'])

        loss_averages.append(epoch_loss['loss'] / batch_per_epoch)
      
        # TODO: don't hardcode this
        torch.save(model, f'/scratch/mr7401/projects/meta_comp/checkpoints/vae_ldim_{latent_dim}_e{epoch}.pt')

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                out = model(batch['image'].to(device))
                test_loss += model.loss_function(*out, M_N=0.00025)['loss'].item()
                if i == 0:
                    n = min(batch['image'].size(0), 8)
                    comparison = torch.cat([batch['image'][:n],
                            out[0].view(batch['image'].shape)[:n].cpu()])
                    os.makedirs(f'/scratch/mr7401/projects/meta_comp/recon/recon_ldim_{latent_dim}', exist_ok=True)
                    save_image(comparison.cpu(),
                               f'/scratch/mr7401/projects/meta_comp/recon/recon_ldim_{latent_dim}/e{str(epoch)}.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
    loss_df = pd.DataFrame({'epoch': range(nepochs), 'loss_avg': loss_averages})
    loss_df.to_csv(f'/scratch/mr7401/projects/meta_comp/checkpoints/vae_ldim_{latent_dim}_losses.csv', index=False)

if __name__ == "__main__":
    app()
