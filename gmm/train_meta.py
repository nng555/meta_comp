from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import torch
import numpy as np
from utils import MetaDataset, gen_collate_fn, SetBatchSampler
from set_transformer2 import SetTransformer2
from torch.optim import AdamW

def build_loader(dataset, batch_size, collate_fn):
    base_sampler = RandomSampler(dataset)
    batch_sampler = SetBatchSampler(batch_size, base_sampler)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

def train(model, nepochs, train_data, test_data, batch_size=32, lr=1e-3, max_samples=30):

    nmodels = train_data.x.shape[1]
    collate_fn = gen_collate_fn(nmodels, batch_size, max_samples)
    optim = AdamW(model.parameters(), lr=lr)

    for epoch in tqdm(range(nepochs)):
        losses = []
        train_loader = build_loader(train_data, batch_size, collate_fn)
        for batch in train_loader:
            optim.zero_grad()
            x1, x2, x_lengths, y = batch
            out = model(x1, x2, x_lengths).squeeze()
            loss = torch.mean((out - torch.abs(y))**2)
            losses.append(loss.item())
            loss.backward()
            optim.step()
        print(np.mean(losses))

if __name__ == "__main__":

    max_samples = 30
    torch.manual_seed(0)
    np.random.seed(0)
    model = SetTransformer2(
        n_inputs=64,
        n_outputs=1,
        n_enc_layers=4,
        dim_hidden=64,
        norm='set_norml',
        sample_size=max_samples,
    )

    train_data = np.load('data/meta_train.npy', allow_pickle=True).item()
    test_data = np.load('data/meta_test.npy', allow_pickle=True).item()

    train_data = MetaDataset(torch.Tensor(train_data['x']), torch.Tensor(train_data['y']))
    test_data = MetaDataset(torch.Tensor(test_data['x']), torch.Tensor(test_data['y']))

    train(model, 500, train_data, test_data, max_samples=max_samples)

