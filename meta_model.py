import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset

class MetaMLP(nn.Module):

    def __init__(
        self,
        hidden_dims=None,
        activation='relu',
    ):
        super(MetaMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [1536, 1000, 1000, 1000, 1]

        if activation == 'relu':
            act_fn = nn.ReLU()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.ReLU())

        self.model = nn.Sequential(*modules[:-1])

    def forward(self, x):
        x = self.model(x)
        return x

class MetaDataset(Dataset):
    def __init__(self, data_dirs):
        self.data = []
        self.labels = []
        self.names = []

        for data_dir in data_dirs:
            for f in os.listdir(data_dir):
                if 'diff.pt' not in f:
                    continue

                self.labels.append(torch.load(os.path.join(data_dir, f), weights_only=True).cuda())
                self.data.append(torch.load(os.path.join(data_dir, f[:-7] + 'img.pt'), weights_only=True).cuda())
                self.names.append(f[:-8])

            self.n_per_set = len(self.data[0])

    def __getitem__(self, idx):
        ridx = idx % self.n_per_set
        sidx = idx // self.n_per_set

        return self.data[sidx][ridx], self.labels[sidx][ridx].float() / 1000.

    def __len__(self):
        return sum([len(d) for d in self.data])

class MetaSubDataset(Dataset):
    def __init__(self, data, labels, name):
        self.name = name
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


