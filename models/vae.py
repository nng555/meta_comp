import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import *
import torchvision.transforms as transforms
from torch.distributions import Normal

IMAGE_SIZE = 150

decode_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.CenterCrop(IMAGE_SIZE)])  # used by decode method to transform final output

class VAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims=None,
                 device='cpu',
                 **kwargs):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules).to(self.device)
        out = self.encoder(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device))
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim).to(self.device)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim).to(self.device)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.size * self.size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules).to(self.device)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())
    
        self.final_layer.to(self.device)

    def encode(self, input):
        input = input.to(self.device)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        z = z.to(self.device)
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.size, self.size)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = decode_transform(result)
        result = torch.nan_to_num(result)
        return result

    def log_likelihood(self, x, n_samples=1000):
        x = x.to(self.device)
        bsize = x.shape[0]
        imsize = np.prod(x.shape[1:])

        with torch.no_grad():
            mu, log_var = self.encode(x)
            log_weights = torch.zeros(bsize, n_samples).to(self.device)

            for k in range(n_samples):
                z = self.reparameterize(mu, log_var)

                log_qz_x = Normal(mu, torch.sqrt(torch.exp(log_var))).log_prob(z).sum(-1)

                log_pz = Normal(0, 1).log_prob(z).sum(-1)

                decoded = self.decode(z)

                log_px_z = -((x.view(bsize, -1) - decoded.view(bsize, -1))**2).sum(-1)

                log_weights[:, k] = log_px_z + log_pz - log_qz_x

        max_log_weights = torch.max(log_weights, dim=1, keepdim=True)[0]
        log_likelihood = (
            max_log_weights.squeeze() +
            torch.log(torch.mean(
                torch.exp(log_weights - max_log_weights), dim=1
            ))
        )

        return log_likelihood

    def reparameterize(self, mu, logvar):
        mu = mu.to(self.device)
        logvar = logvar.to(self.device)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return eps * std + mu

    def forward(self, input, **kwargs):
        input = input.to(self.device)
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        recons = args[0].to(self.device)
        input = args[1].to(self.device)
        mu = args[2].to(self.device)
        log_var = args[3].to(self.device)

        bsize = input.shape[0]

        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons.view(bsize, -1), input.view(bsize, -1))

        kld_loss = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples,
               current_device):
        z = torch.randn(num_samples,
                        self.latent_dim).to(self.device)

        log_z = Normal(
            torch.zeros_like(z),
            torch.ones_like(z)
        ).log_prob(z).sum(-1)

        sample_means = torch.clamp(
            self.decode(z),
            min=0., max=1.,
        )

        return sample_means, log_z

    def generate(self, x, **kwargs):
        x = x.to(self.device)
        return self.forward(x)[0]
