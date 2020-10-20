import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from .ResidualBlocks import ResidualBlockEncoder, ResidualBlockDecoder
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')


class VAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 in_size: int, 
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.nb_last_channels = hidden_dims[-1]
        self.out_channels = in_channels
        

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            div = 2**5  # Model reduces size by this factor           
        else:
            div = 2**len(hidden_dims)   # Model reduces size by this factor   
            
        # Make sure input dimension is usable     
        if in_size%div==0:
            self.smallest_size = int(in_size/div)
        else: raise ValueError('Input size not compatible with number of hidden layers.')


        self.latent_dim = hidden_dims[-1] * self.smallest_size**2


            

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    ResidualBlockEncoder(in_channels=in_channels, out_channels=in_channels, stride= 1),
                    ResidualBlockEncoder(in_channels=in_channels, out_channels=in_channels, stride= 1),
                    ResidualBlockEncoder(in_channels=in_channels, out_channels=h_dim, stride= 2))
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules) 
        self.fc_mu = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size= 3, stride= 1, padding  = 1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size= 3, stride= 1, padding  = 1)
        


        # Build Decoder
        modules = []

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    ResidualBlockDecoder(hidden_dims[i], hidden_dims[i], stride = 1),
                    ResidualBlockDecoder(hidden_dims[i], hidden_dims[i], stride = 1),
                    ResidualBlockDecoder(hidden_dims[i], hidden_dims[i+1], stride = 2))
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.out_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution       
        mu = torch.flatten(self.fc_mu(result), start_dim=1)
        log_var = torch.flatten(self.fc_var(result), start_dim=1)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = z
        result = result.view(-1, self.nb_last_channels, self.smallest_size, self.smallest_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
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

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
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

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]