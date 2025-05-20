import torch
from einops import rearrange
import itertools
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torch.nn.functional as F

class OlDMetaDataset(Dataset):

    def __init__(self, meta_x, meta_y):
        self.x = meta_x # N x M x D
        self.y = meta_y # N x M x M

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] # M x D, M x M

def gen_collate_fn(nmodels, batch_size, max_samples):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def collate_fn(batch): # B x 2 x N

        # ragged tensors
        batch_x = [nset[0] for nset in batch] # B x N x M x D
        x_lengths = [len(x) for x in batch_x]
        batch_y = [nset[1] for nset in batch] # B x N x M x M

        # right pad to max_samples with zeros
        batch_x = [F.pad(bx, (0, 0, 0, 0, 0, max_samples - bl), 'constant', 0) for bx, bl in zip(batch_x, x_lengths)]
        batch_x = torch.stack(batch_x)

        # choose models to compare
        model_select = np.random.randint(0, high=nmodels, size=batch_size)
        model_offset = np.random.randint(1, high=nmodels - 1, size=batch_size)
        model_select2 = (model_select + model_offset) % nmodels

        # grab input data and generate KL diff label
        batch_idxs = torch.arange(batch_size)
        batch_x1 = batch_x[batch_idxs, :, model_select]
        batch_x2 = batch_x[batch_idxs, :, model_select2]

        # maybe separate into magnitude and direction?
        batch_y = [(by[:, ms, ms2] - by[:, ms2, ms]).mean() for (by, ms, ms2) in zip(batch_y, model_select, model_select2)]
        batch_y = torch.Tensor(batch_y)

        x_lengths = torch.Tensor(x_lengths).to(device)
        batch_y = batch_y.to(device)

        return batch_x1, batch_x2, x_lengths, batch_y

    return collate_fn

def interleave_batch(x1, x2):
    return  rearrange([x1, x2], 't b n d -> b (n t) d')

def uninterleave_batch(x):
    return x[:, ::2], x[:, 1::2]

class SetBatchSampler(BatchSampler):

    def __init__(self, batch_size, sampler, min_samples=20, max_samples=30):
        super(SetBatchSampler, self).__init__(sampler, batch_size, drop_last=True)
        self.min_samples = min_samples
        self.max_samples = max_samples

    def get_nsamples(self):
        return np.random.randint(self.min_samples, high=self.max_samples + 1, size=self.batch_size)

    def __iter__(self):
        sampler_iter = iter(self.sampler)

        nsamples = self.get_nsamples()

        # TODO: sample 2 different sizes for each batch input
        batch = [[*itertools.islice(sampler_iter, n)] for n in nsamples]
        while batch[-1]:
            yield batch
            nsamples = self.get_nsamples()
            batch = [[*itertools.islice(sampler_iter, n)] for n in nsamples]

def mask_matrix(matrix, lengths):
    """
    lengths is [batch, 1]

    """
    if lengths is None:
        return matrix
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert lengths.shape == (matrix.shape[0], 1), f"{lengths.shape} vs. {(matrix.shape[0], 1)}"
    batch, n_samples, n_feats = matrix.shape
    # [batch, n_samples]
    length_mask = torch.arange(n_samples).expand(batch, n_samples).to(device) < lengths
    return matrix * length_mask.unsqueeze(-1)


def reshape_x_and_lengths(x, lengths, device):
    x = x.to(device)
    if lengths is None:
        # print('creating lengths')
        lengths = x.shape[1] * torch.ones((x.shape[0], 1)).to(device)
    else:
        lengths = lengths.reshape(x.shape[0], 1)
    assert lengths.shape == (x.shape[0],1), f"lengths should be shaped [batch, n_dist]: {lengths.shape} vs. {(x.shape[0],1)}"
    return x, lengths


def aggregation(x, lengths, input_shape, device, type='mean', categorical=False):
    """
    x: [batch, sample_size, hidden_units]
    (due to the concatenation of the individual encoder outputs)
    lengths: [batch, 1]

    """
    if categorical:
        batch, n_samples = input_shape
    else:
        batch, n_samples, _ = input_shape
    x = x.reshape(batch, n_samples, -1)
    # [batch, n_samples]
    length_mask = torch.arange(n_samples).expand(lengths.shape[0], 1, n_samples).to(device) < lengths.unsqueeze(-1)
    length_mask = length_mask.squeeze()
    if type == 'sum':
        out = (x * length_mask.unsqueeze(-1)).sum(dim=-2)
    elif type == 'mean':
        # numerator is [batch, n_dists, hidden_units]
        # denominator is [batch, n_dists, 1]
        out = (x * length_mask.unsqueeze(-1)).sum(dim=-2) / length_mask.sum(dim=-1).unsqueeze(-1)
    elif type == 'max':
        length_mask = (1-length_mask.type(torch.FloatTensor).to(device))#*
        length_mask[length_mask!=0] = -float("Inf")
        out = (x+length_mask.unsqueeze(-1)).max(dim=-2)[0]
    else:
        raise ValueError(f"Unsupported type aggregation: {type}")

    out = out.reshape(batch, -1)
    assert len(out.shape) == 2

    if torch.all(torch.eq(lengths, n_samples)):
        if type == 'mean':
            assert torch.allclose(out, x.mean(dim=1).reshape(batch, -1), rtol=1e-05, atol=1e-05), f"aggregation is off: {out} vs. {x.mean(dim=2).reshape(batch, -1)}"
        elif type == 'sum':
            assert torch.allclose(out, x.sum(dim=1).reshape(batch, -1), rtol=1e-05, atol=1e-05), f"aggregation is off: {out} vs. {x.sum(dim=2).reshape(batch, -1)}"
        elif type == 'max':
            assert torch.allclose(out, x.max(dim=1)[0].reshape(batch, -1), rtol=1e-05, atol=1e-05), f"aggregation is off: {out} vs. {x.max(dim=1).reshape(batch, -1)}"
    return out


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                print('len(inputs)', len(inputs))
                inputs = module(inputs)
        return inputs

class MyLinear(nn.Linear):
    def forward(self, x, lengths):
        return super().forward(x), lengths

class MySigmoid(nn.Sigmoid):
    def forward(self, x, lengths):
        return super().forward(x), lengths
