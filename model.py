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
                 **kwargs):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)

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

        self.decoder = nn.Sequential(*modules)

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
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.size, self.size)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = decode_transform(result)
        #result = torch.flatten(result, start_dim=1)
        result = torch.nan_to_num(result)
        return result

    def log_likelihood(self, x, n_samples=1000):
        bsize = x.shape[0]
        imsize = np.prod(x.shape[1:])

        with torch.no_grad():
            mu, log_var = self.encode(x)
            log_weights = torch.zeros(bsize, n_samples).cuda()

            for k in range(n_samples):
                z = self.reparameterize(mu, log_var)

                log_qz_x = Normal(mu, torch.sqrt(torch.exp(log_var))).log_prob(z).sum(-1)

                log_pz = Normal(0, 1).log_prob(z).sum(-1)

                decoded = self.decode(z)

                # assume normal distribution since we trained with MSE loss on reconstruction
                log_px_z = -((x.view(bsize, -1) - decoded.view(bsize, -1))**2).sum(-1)
                # these are constants for all images
                    #- np.log(sigma) * imsize
                    #- 0.5 * np.log(2 * np.pi) * imsize

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
        recons_loss = F.mse_loss(recons.view(bsize, -1), input.view(bsize, -1))

        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

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

        log_z = Normal(
            torch.zeros_like(z),
            torch.ones_like(z)
        ).log_prob(z).sum(-1)

        sample_means = torch.clamp(
            self.decode(z),
            min=0., max=1.,
        )

        """
        noise = torch.normal(
            mean=torch.zeros_like(sample_means),
            std=torch.ones_like(sample_means) * np.sqrt(0.5),
        )

        samples = torch.clamp(
            sample_means + noise,
            min=0, max=1,
        )
        """

        """
        log_x_z = Normal(
            torch.zeros_like(noise),
            torch.ones_like(noise) * np.sqrt(0.5)
        ).log_prob(noise).sum([-1, -2, -3])

        samples = sample_means + noise
        likelihoods = log_z + log_x_z
        """

        return sample_means, log_z

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
