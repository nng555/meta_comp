from tqdm import tqdm
import sys
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModel, ViTImageProcessor

import torch
from model import VAE
import os
from utils import *

import typer
app = typer.Typer()

@app.command()
def gen_ll(
    mpath: str=None,
    model_name: str=None,
    sample_path: str=None,
    sample_name: str=None,
    bsize: int=100,
    n_ll_samples: int=200,
):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model = torch.load(mpath, weights_only=False)
        model.to(device).eval()

        samples = torch.load(sample_path)
        nsamples = len(samples)
        nbatches = nsamples // bsize

        logls = []

        for i in tqdm(range(nbatches)):
            bstart = i * bsize
            bend = (i + 1) * bsize
            logls.append(model.log_likelihood(samples[bstart:bend], n_samples=n_ll_samples))

        logls = torch.concatenate(logls)

        save_dir = f"generations/{sample_name}"
        os.makedirs(save_dir, exist_ok=True)

        torch.save(logls, f"generations/{sample_name}/{model_name}_ll.pt")

if __name__ == "__main__":
    app()
