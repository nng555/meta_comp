import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys 
sys.path.append("/scratch/mr7401/projects/meta_comp/")
sys.path.append("/scratch/mr7401/projects/meta_comp/gmm/")
from .norms import get_norm
from .utils import mask_matrix, reshape_x_and_lengths, MySequential, MyLinear, interleave_batch, uninterleave_batch, MyReLU

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads,
                 norm='ln1_fp', v_norm_samples=32,
                 sample_size=1000, normalizeQ=True):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.normalizeQ = normalizeQ
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.normQ = get_norm(norm, sample_size=sample_size, dim_V=dim_Q)
        self.normK = get_norm(norm, sample_size=v_norm_samples, dim_V=dim_K)
        self.norm0 = get_norm(norm, sample_size=sample_size, dim_V=dim_V)

    def forward(self, Q, K, lengths=None, mask=[]):
        _input = Q
        if self.normalizeQ:
            Q = Q if getattr(self, 'normQ', None) is None else self.normQ(Q, lengths)[0]
        K = K if getattr(self, 'normK', None) is None else self.normK(K, lengths)[0]

        Q = self.fc_q(Q)
        if "Q" in mask:
            Q = mask_matrix(Q, lengths)

        K, V = self.fc_k(K), self.fc_v(K)
        if "K" in mask:
            K = mask_matrix(K, lengths)
            V = mask_matrix(V, lengths)

        dim_split = self.dim_V // self.num_heads
        dim_split = 2**int(round(np.log2(dim_split),0))
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        input_multihead = torch.cat(_input.split(dim_split, 2), 0)
        O = torch.cat((input_multihead + A.bmm(V_)).split(Q.size(0), 0), 2)
        normed_O = O if getattr(self, 'norm0', None) is None else self.norm0(O, lengths)[0]
        O = O + F.relu(self.fc_o(normed_O))
        return O


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, sample_size=100, norm="none"):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, norm=norm,
                        v_norm_samples=sample_size, sample_size=num_inds,
                        normalizeQ=False)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, norm=norm,
                        v_norm_samples=num_inds, sample_size=sample_size,
                        normalizeQ=True)

    def forward(self, X, lengths=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, lengths=lengths, mask=["K"])
        return self.mab1(X, H, lengths=lengths, mask=["Q"]), lengths

class XISAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, num_inds, sample_size=100, norm="none"):
        super(XISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))

        nn.init.xavier_uniform_(self.I)

        # query with I -> X
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, norm=norm,
                          v_norm_samples=sample_size, sample_size=num_inds,
                          normalizeQ=False)

        # query with IY -> IX
        self.mabx = MAB(dim_out, dim_out, dim_out, num_heads, norm=norm,
                        v_norm_samples=sample_size, sample_size=num_inds,
                        normalizeQ=True)

        # query with X -> IX
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, norm=norm,
                        v_norm_samples=num_inds, sample_size=sample_size,
                        normalizeQ=True)

    def forward(self, X, Y, xlengths, ylengths):
        HX = self.mab0(self.I.repeat(X.size(0), 1, 1), X, lengths=xlengths, mask=["K"])
        HY = self.mab0(self.I.repeat(Y.size(0), 1, 1), Y, lengths=ylengths, mask=["K"])

        HX, HX_lengths = reshape_x_and_lengths(HX, None, torch.device("cuda"))
        HY, HY_lengths = reshape_x_and_lengths(HX, None, torch.device('cuda'))

        HX = self.mabx(HX, HY, lengths=HX_lengths)
        HY = self.mabx(HY, HX, lengths=HY_lengths)

        HX = self.mab1(X, HX, lengths=xlengths, mask=["Q"])
        HY = self.mab1(Y, HY, lengths=ylengths, mask=["Q"])

        return HX, HY, xlengths, ylengths


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, sample_size=1, norm='none'):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads,
                        sample_size=sample_size, norm=norm)

    def forward(self, X, lengths=None):
        return self.mab(X, X, lengths=lengths, mask=['Q', 'K']), lengths


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, norm='none', sample_size=1)

    def forward(self, X, lengths=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, lengths=lengths, mask=["K"]), lengths

# encode using the same tower then late fusion with SAB
class SetTransformer2(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000):
        super(SetTransformer2, self).__init__()

        num_heads = 8
        num_inds = 128

        self.in_proj = nn.Sequential(
            nn.Linear(n_inputs, dim_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_hidden, dim_hidden),
        )

        layers = []

        for i in range(n_enc_layers):
            #layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, norm=norm, sample_size=sample_size))
            layers.append(SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=sample_size))
        if norm != "none":
            layers.append(get_norm(norm, sample_size=sample_size, dim_V=dim_hidden))

        self.enc = MySequential(*layers)
        self.out_proj = PMA(dim_hidden, num_heads, 1)
        self.dec = MySequential(
            SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=2),
            SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=2),
            SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=2),
        )
        self.mag_head = MySequential(
            PMA(dim_hidden, num_heads, 1),
            MyLinear(dim_hidden, dim_hidden),
            MyReLU(),
            MyLinear(dim_hidden, 1),
        )
        self.dir_head = nn.Sequential(
            nn.Linear(2 * dim_hidden, 2 * dim_hidden),
            nn.ReLU(),
            nn.Linear(2 * dim_hidden, 1),
        )

    def forward(self, x1, x2, x_lengths):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x1, x1_lengths = reshape_x_and_lengths(x1, x_lengths, device)
        x2, x2_lengths = reshape_x_and_lengths(x2, x_lengths, device)

        x1 = self.in_proj(x1)
        x2 = self.in_proj(x2)
        #x1, x1_lengths = self.in_proj(x1, x1_lengths)
        #x2, x2_lengths = self.in_proj(x2, x2_lengths)
        x1_out, _ = self.enc(x1, x1_lengths)
        x2_out, _ = self.enc(x2, x2_lengths)
        #x1_out, x2_out, _, _= self.enc(x1, x2, x1_lengths, x2_lengths)
        x1_out, _ = self.out_proj(x1_out, x1_lengths)
        x2_out, _ = self.out_proj(x2_out, x2_lengths)

        x = torch.concatenate((x1_out, x2_out), 1)
        x, x_lengths = reshape_x_and_lengths(x, None, device)
        x, x_lengths = self.dec(x, x_lengths)

        out_mag = F.relu(self.mag_head(x, x_lengths)[0])
        x = torch.flatten(x, start_dim=1)
        out_dir = self.dir_head(x)

        return out_mag, out_dir

# encode everything together then separately aggregate and combine with SAB
class SetTransformer3(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000):
        super(SetTransformer3, self).__init__()

        num_heads = 4
        num_inds = 32

        self.in_proj = MyLinear(n_inputs, dim_hidden)
        layers = []

        for i in range(n_enc_layers):
            layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, norm=norm, sample_size=sample_size * 2))
        self.enc = MySequential(*layers)
        self.out_proj = PMA(dim_hidden, num_heads, 1)
        self.dec = MySequential(
            SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=2),
            SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=2),
            PMA(dim_hidden, num_heads, 1),
            MyLinear(dim_hidden, n_outputs),
        )
        self.max_samples = sample_size

    def forward(self, x1, x2, x_lengths):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = interleave_batch(x1, x2)
        x, nx_lengths = reshape_x_and_lengths(x, 2 * x_lengths, device)
        x, nx_lengths = self.in_proj(x, nx_lengths)
        x, _ = self.enc(x, nx_lengths)

        # separate and aggregate
        x1, x2 = uninterleave_batch(x)

        x1, x1_lengths = reshape_x_and_lengths(x1, x_lengths, device)
        x2, x2_lengths = reshape_x_and_lengths(x2, x_lengths, device)

        x1_out, _ = self.out_proj(x1, x1_lengths)
        x2_out, _ = self.out_proj(x2, x2_lengths)

        x = torch.concatenate((x1_out, x2_out), 1)
        x, x_lengths = reshape_x_and_lengths(x, None, device)
        out, _ = self.dec(x, x_lengths)
        return out

# encode using the same tower then late fusion with MAB
class SetTransformer4(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000):
        super(SetTransformer4, self).__init__()

        num_heads = 4
        num_inds = 32

        self.in_proj = MyLinear(n_inputs, dim_hidden)
        layers = []

        for i in range(n_enc_layers):
            layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, norm=norm, sample_size=sample_size))
        #if norm != "none":
        #    layers.append(get_norm(norm, sample_size=sample_size, dim_V=dim_hidden))
        self.enc = MySequential(*layers)

        self.transition = MAB(dim_hidden, dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=sample_size, v_norm_samples=sample_size)

        layers2 = []
        for i in range(2):
            layers2.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, norm=norm, sample_size=sample_size))
        self.enc2 = MySequential(*layers2)

        self.dec = MySequential(
            PMA(dim_hidden, num_heads, 1),
            MyLinear(dim_hidden, n_outputs),
        )
        self.max_samples = sample_size

    def forward(self, x1, x2, x_lengths):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x1, x1_lengths = reshape_x_and_lengths(x1, x_lengths, device)
        x2, x2_lengths = reshape_x_and_lengths(x2, x_lengths, device)

        x1, x1_lengths = self.in_proj(x1, x1_lengths)
        x2, x2_lengths = self.in_proj(x2, x2_lengths)
        x1_out, _ = self.enc(x1, x1_lengths)
        x2_out, _ = self.enc(x2, x2_lengths)

        x = self.transition(x1, x2, lengths=x1_lengths, mask=["Q", "K"])
        x, x_lengths = self.enc2(x, x1_lengths)

        #x1_out, x2_out, _, _= self.enc(x1, x2, x1_lengths, x2_lengths)
        out, _ = self.dec(x, x1_lengths)
        return out

# encode using the same tower then late fusion with SAB
class SetTransformer5(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000):
        super(SetTransformer5, self).__init__()

        num_heads = 4
        num_inds = 32

        self.in_proj = MyLinear(n_inputs, dim_hidden)
        layers = []
        for i in range(n_enc_layers):
            layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, norm=norm, sample_size=sample_size))
        #if norm != "none":
        #    layers.append(get_norm(norm, sample_size=sample_size, dim_V=dim_hidden))
        self.enc = MySequential(*layers)

        self.enc2 = MySequential(*[
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, norm=norm, sample_size=sample_size * 2) for _ in range(n_enc_layers)
        ])

        self.out_proj = PMA(dim_hidden, num_heads, 1)
        self.dec = MySequential(
            SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=2),
            SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=2),
        )
        self.mag_head = MySequential(
            PMA(dim_hidden, num_heads, 1),
            MyLinear(dim_hidden, 1),
        )
        self.dir_head = nn.Sequential(
            nn.Linear(2 * dim_hidden, 2 * dim_hidden),
            nn.ReLU(),
            nn.Linear(2 * dim_hidden, 1),
        )

    def forward(self, x1, x2, x_lengths):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x1, x1_lengths = reshape_x_and_lengths(x1, x_lengths, device)
        x2, x2_lengths = reshape_x_and_lengths(x2, x_lengths, device)

        x1, x1_lengths = self.in_proj(x1, x1_lengths)
        x2, x2_lengths = self.in_proj(x2, x2_lengths)
        x1_out, _ = self.enc(x1, x1_lengths)
        x2_out, _ = self.enc(x2, x2_lengths)

        x = interleave_batch(x1_out, x2_out)
        x, nx_lengths = reshape_x_and_lengths(x, 2 * x_lengths, device)
        x, _ = self.enc2(x, nx_lengths)

        x1_out, x2_out = uninterleave_batch(x)

        x1_out, _ = self.out_proj(x1_out, x1_lengths)
        x2_out, _ = self.out_proj(x2_out, x2_lengths)

        x = torch.concatenate((x1_out, x2_out), 1)
        x, x_lengths = reshape_x_and_lengths(x, None, device)
        x, x_lengths = self.dec(x, x_lengths)

        out_mag = F.relu(self.mag_head(x, x_lengths)[0])
        x = torch.flatten(x, start_dim=1)
        out_dir = self.dir_head(x)

        return out_mag, out_dir
