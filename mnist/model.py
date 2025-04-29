import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import *
import torchvision.transforms as transforms
from torch.distributions import Normal

IMAGE_SIZE = 28

class VAE(nn.Module):

    def __init__(self,
                 in_dim,
                 latent_dim,
                 hidden_dims=None,
                 **kwargs):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [512]

        # Build Encoder
        tmp_dim = in_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(tmp_dim, h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            tmp_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        modules = []
        hidden_dims.reverse()
        tmp_dim = latent_dim

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(tmp_dim, h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            tmp_dim = h_dim

        modules.append(nn.Linear(tmp_dim, in_dim))

        # clamp to 0-1 pixel values
        modules.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        result = torch.nan_to_num(result)
        return result

    def log_likelihood(self, x, n_samples=1000):
        bsize = x.shape[0]

        with torch.no_grad():
            mu, log_var = self.encode(x)
            log_weights = torch.zeros(bsize, n_samples).cuda()

            for k in range(n_samples):
                z = self.reparameterize(mu, log_var)

                log_qz_x = Normal(mu, torch.exp(0.5 * log_var)).log_prob(z).sum(-1)

                log_pz = Normal(0, 1).log_prob(z).sum(-1)

                decoded = self.decode(z)

                # use bernoulli distribution log likelihood calculation
                #log_px_Z = F.binary_cross_entropy(decoded, x, reduction='none').sum(-1)
                log_px_z = -((x - decoded)**2).sum(-1)

                # importance sampling
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
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        bsize = input.shape[0]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        #recons_loss = F.binary_cross_entropy(recons, input)
        recons_loss = F.mse_loss(recons, input, reduction='sum') / bsize

        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()) / bsize

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'recon_loss': recons_loss.detach(), 'kld': -kld_loss.detach()}

    def sample(self,
               num_samples,
               current_device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        sample_means = torch.clamp(
            self.decode(z),
            min=0., max=1.,
        )

        return sample_means

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
