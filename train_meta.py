from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import torch
import numpy as np
from mnist.utils import MetaDataset, gen_collate_fn, SetBatchSampler
from gmm.set_transformer2 import *
from torch.optim import AdamW
import wandb
np.set_printoptions(suppress=True)

wandb.login()
run = wandb.init(project="nts_mnist")

def build_loader(dataset, batch_size, collate_fn):
    rs = RandomSampler(dataset)
    bs = SetBatchSampler(batch_size, rs, dataset.folder_names, min_samples=20, max_samples=30)
    loader = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_fn)
    return loader

def train(model, nepochs, train_data, test_data, batch_size=32, lr=1e-3, max_samples=30, loss_fn='mse_dir', temperature=1.0):

    collate_fn = gen_collate_fn(max_samples)
    optim = AdamW(model.parameters(), lr=lr)

    for epoch in tqdm(range(nepochs)):
        losses = []
        train_loader = build_loader(train_data, batch_size, collate_fn)
        for batch in train_loader:
            optim.zero_grad()
            x1, x2, x_lengths, y = batch
            out = model(x1, x2, x_lengths)
            if loss_fn == 'mse':
                out = out.squeeze()
                loss = torch.mean((F.relu(out) - torch.abs(y))**2)
            elif loss_fn == 'mse_dir':
                mag_loss = torch.mean((out[0].squeeze() - torch.abs(y))**2)
                dir_loss = F.binary_cross_entropy_with_logits(out[1].squeeze(), (y > 0).float())
                loss = 100 * mag_loss + dir_loss
            elif loss == "bce":
                out = out.squeeze()
                out /= temperature
                y /= temperature
                py = F.sigmoid(y)
                loss = F.binary_cross_entropy_with_logits(out, py)
            losses.append(loss.item())
            if loss_fn == 'mse_dir':
                wandb.log({'mse_loss': mag_loss.item(),
                           'bce_loss': dir_loss.item(),
                           'bce_acc': ((out[1].squeeze() > 0) == (y > 0)).float().mean(),
                           'mse_dir_loss': loss.item(),
                })
                out = out[0].squeeze() * ((out[1].squeeze() > 0).float() * 2 - 1)
            else:
                wandb.log({loss_fn + "_loss": loss.item()})
            loss.backward()
            optim.step()
        if (epoch + 1) % 100 == 0:
            print(out.detach().cpu().numpy())
            print(y.cpu().numpy())
        #print(np.mean(losses))

    torch.save(model, 'st5_4l.ckpt')


if __name__ == "__main__":
    max_samples = 30
    torch.manual_seed(0)
    np.random.seed(0)
    model = SetTransformer2(
        n_inputs=784,
        n_outputs=1,
        n_enc_layers=4,
        dim_hidden=512,
        norm='set_norml',
        sample_size=max_samples,
    )

    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
        model.to(device)
    else:
        device = torch.device('cpu')

    train_data = MetaDataset('mnist/generations/train')
    test_data = MetaDataset('mnist/generations/test')

    train(model, 1000, train_data, test_data, max_samples=max_samples, loss_fn='mse_dir', temperature=1)

