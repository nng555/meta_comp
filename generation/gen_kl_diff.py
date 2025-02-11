from tqdm import tqdm
from torchvision.utils import save_image
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModel, ViTImageProcessor

import torch
from model import VAE
import os

import typer
app = typer.Typer()

def get_embeddings(model, extractor, imgs, device):
    inputs = extractor(
        images=[x.detach().cpu().numpy() for x in imgs], return_tensors="pt"
    ).to(device)
    embeds = model(**inputs).last_hidden_state[:, 0].cpu()
    return embeds

@app.command()
def gen_kl_diff(
    m1_path: str=None,
    m2_base_path: str=None,
    bsize: int=100,
    n_kl_samples: int=10000,
    n_ll_samples: int=100,
    embed: bool=True,
    embed_model_name: str="google/vit-base-patch16-224",
):
    with torch.no_grad():
        if embed:
            extractor = ViTImageProcessor.from_pretrained(embed_model_name, do_rescale=False)
            embed_model = AutoModel.from_pretrained(embed_model_name, output_hidden_states=True)
            embed_model.cuda()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m1 = torch.load(m1_path, weights_only=False)
        m1.to(device).eval()

        m1_name = os.path.basename(m1_path)[:-3]

        m1_parent_dir = str(Path(m1_path).parent.stem)
        m2_parent_dir = os.path.basename(m2_base_path)

        save_dir = '/scratch/nhn234/data/kl_diffs/' + m1_parent_dir + '_vs_' + m2_parent_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for m2_path in os.listdir(m2_base_path):

            m2 = torch.load(
                os.path.join(m2_base_path, m2_path),
                weights_only=False
            )

            save_path = os.path.join(save_dir, m1_name + '_vs_' + m2_path)[:-3]
            print(save_path, flush=True)
            m2.to(device).eval()

            kl_diffs = []
            m1_full_samples = []
            m2_full_samples = []

            nbatches = n_kl_samples // bsize

            for i in tqdm(range(nbatches), total=nbatches):

                m1_samples, m1_m1_ll = m1.sample(bsize, device)
                m2_samples, m2_m2_ll = m2.sample(bsize, device)

                m1_m2_ll = m1.log_likelihood(m2_samples, n_samples=n_ll_samples)
                m2_m1_ll = m2.log_likelihood(m1_samples, n_samples=n_ll_samples)

                kl_diffs.append(-(m1_m1_ll + m1_m2_ll) + (m2_m1_ll + m2_m2_ll))

                if embed:
                    m1_full_samples.append(get_embeddings(embed_model, extractor, m1_samples, device))
                    m2_full_samples.append(get_embeddings(embed_model, extractor, m2_samples, device))
                else:
                    m1_full_samples.append(m1_samples)
                    m2_full_samples.append(m2_samples)

            kl_diffs = torch.concatenate(kl_diffs)
            m1_samples = torch.concatenate(m1_full_samples)
            m2_samples = torch.concatenate(m2_full_samples)
            samples = torch.concatenate((m1_samples, m2_samples), axis=-1)
            torch.save(kl_diffs, save_path + '_diff.pt')
            torch.save(samples, save_path + '_img.pt')

        #diffs = kl_diffs.view(n_mean_samples, -1).mean(0).cpu().numpy()

        #sns.kdeplot(diffs)
        #plt.savefig('/home/nhn234/test.png')
        #import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    app()
