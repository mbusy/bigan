import torch
from torch import nn
from torch.nn import functional as F


class Unsqueeze(nn.Module):
    def forward(self, input):
        return input.unsqueeze(2).unsqueeze(3)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BaseModule(nn.Module):
    """
    Base module for the Bidirectional Generative Adversarial Network
    """
    def __init__(self, channels=1, width=28, height=28, z_dim=50):
        super(BaseModule, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.z_dim = z_dim
        self.h_dim = 1024
        self.slope = 1e-2
        self.dropout = 0.2
        self.discriminator_output_size = 1

    def get_nb_channels(self):
        """
        Returns the number of channels of the handled images
        """
        return self.channels

    def get_width(self):
        """
        Returns the width of the handled images in pixels
        """
        return self.width

    def get_height(self):
        """
        Returns the height of the handled images in pixels
        """
        return self.height

    def get_z_dim(self):
        """
        Returns the dimension of the latent space of the VAE
        """
        return self.z_dim

    def flatten(self, x):
        """
        Can be used to flatten the output image. This method will only handle
        images of the original size specified for the network
        """
        return x.view(-1, self.channels * self.height * self.width)

    def forward(self, input):
        raise NotImplementedError


class Generator(BaseModule):
    """
    Generator of the Bidirectional Generative Adversarial Network
    """
    def __init__(self, channels=1, width=28, height=28, z_dim=50):
        """
        Constructor

        Parameters:
            channels - The number of channels in the image
            width - The width of the image in pixels
            height - The height of the image in pixels
            z_dim - The dimension of the latent space (output of the encoder)
        """
        super(Generator, self).__init__(channels, width, height, z_dim)
        self.generator = nn.Sequential(
            Unsqueeze()
            # input dim: z_dim x 1 x 1
            nn.ConvTranspose2d(self.z_dim, 256, 4, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim:   256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 128 x 10 x 10
            nn.ConvTranspose2d(128, 64, 4, stride=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 64 x 13 x 13
            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 32 x 28 x 28
            nn.ConvTranspose2d(32, self.channels, 1, stride=1, bias=True), # Conv ?
            # output dim: channels x 28 x 28
            nn.Sigmoid())

        def forward(self, input):
            return self.generator(input)


class Encoder(BaseModule):
    """
    Encoder of the Bidirectional Generative Adversarial Network
    """
    def __init__(self, channels=1, width=28, height=28, z_dim=50):
        """
        Constructor

        Parameters:
            channels - The number of channels in the image
            width - The width of the image in pixels
            height - The height of the image in pixels
            z_dim - The dimension of the latent space (output of the encoder)
        """
        super(Encoder, self).__init__(channels, width, height, z_dim)
        self.encoder = nn.Sequential(
            # input dim: channels x 32 x 32
            nn.Conv2d(self.channels, 32, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 32 x 28 x 28
            nn.Conv2d(32, 64, 4, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 64 x 13 x 13
            nn.Conv2d(64, 128, 4, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 128 x 10 x 10
            nn.Conv2d(128, 256, 4, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 256 x 4 x 4
            nn.Conv2d(256, 512, 4, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 512 x 1 x 1
            nn.Conv2d(512, 512, 1, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, inplace=True),
            # state dim: 512 x 1 x 1
            # output dim: opt.z_dim x 1 x 1
            nn.Conv2d(512, z_dim, 1, stride=1, bias=True))

    def forward(self, input):
        return self.encoder(input)


class Discriminator(BaseModule):
    """
    Discriminator of the Bidirectional Generative Adversarial Network
    """
    def __init__(self, channels=1, width=28, height=28, z_dim=50):
        """
        Constructor

        Parameters:
            channels - The number of channels in the image
            width - The width of the image in pixels
            height - The height of the image in pixels
            z_dim - The dimension of the latent space (output of the encoder)
        """
        super(Discriminator, self).__init__(channels, width, height, z_dim)
        self.discriminator_infer_x = nn.Sequential(
            # state dim: channels 28 x 28
            nn.Conv2d(self.channels, 64, 4, stride=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Dropout2d(p=self.dropout),
            # state dim: 64 x 13 x 13
            nn.Conv2d(64, 128, 4, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Dropout2d(p=self.dropout),
            # state dim: 128 x 10 x 10
            nn.Conv2d(128, 256, 4, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Dropout2d(p=self.dropout),
            # state dim: 256 x 4 x 4
            nn.Conv2d(256, 512, 4, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Dropout2d(p=self.dropout),
            # output dim: 512 x 1 x 1
            Flatten())

        self.discriminator_infer_z = Flatten()

        self.discriminator_infer_joint = nn.Sequential(
            torch.nn.Linear(512 + self.z_dim, self.h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(self.h_dim, 1),
            torch.nn.Sigmoid())

    def forward(self, input_x, input_z):
        output_x = self.discriminator_infer_x(input_x)
        output_z = self.discriminator_infer_z(input_z)
        return self.discriminator_infer_joint(torch.cat(
            [output_x, output_z],
            dim=1))


class BiGAN(object):
    """
    Meta class defining a Bidirectional Generative Adversarial Network
    """
    def __init__(self, device, channels=1, width=28, height=28, z_dim=50):
        """
        Constructor

        Parameters:
            device - "cuda" or "cpu", device on which the network is deployed
            channels - The number of channels in the image
            width - The width of the image in pixels
            height - The height of the image in pixels
            z_dim - The dimension of the latent space (output of the encoder)
        """
        self.generator = Generator(
            channels=channels,
            width=width,
            height=height,
            z_dim=z_dim).to(device)

        self.encoder = Encoder(
            channels=channels,
            width=width,
            height=height,
            z_dim=z_dim).to(device)

        self.discriminator = Discriminator(
            channels=channels,
            width=width,
            height=height,
            z_dim=z_dim).to(device)
