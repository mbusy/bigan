import os
import torch
import numpy as np
from tqdm import tqdm
from itertools import *
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bigan import BiGAN


class NetManager:
    """
    Net manager for the Bidirectional Generative Adversarial Network
    """

    def __init__(
            self,
            bigan,
            device,
            train_loader=None,
            test_loader=None,
            lr=1e-3):
        """
        Constructor
        """
        self.bigan = bigan
        self.device = device
        self.writer = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.weight_decay = 2.5*1e-5

        parameters_g = chain(
            self.bigan.encoder.parameters(),
            self.bigan.generator.parameters())

        self.optimizer_g = optim.Adam(
            parameters_g,
            lr=self.lr
            betas=[self.beta_1, self.beta_2],
            weight_decay=self.weight_decay)

        self.optimizer_d = optim.Adam(
            self.bigan.discriminator.parameters(),
            lr=self.lr
            betas=[self.beta_1, self.beta_2],
            weight_decay=self.weight_decay)

    def set_writer(self, board_name):
        """
        Sets a torch writer object. The logs will be generated in logs/name
        """
        if isinstance(self.writer, SummaryWriter):
            self.writer.close()

        if board_name is None:
            self.writer = None
        else:
            self.writer = SummaryWriter("logs/" + board_name)

    def load_net(self, network_state_name):
        """
        Loads a model
        """
        network_state_dict = torch.load(network_state_name)
        self.model.load_state_dict(network_state_dict)

    def save_net(self, path):
        """
        Saves the net currently loaded in the manager, at the specified path
        """
        torch.save(
            self.bigan.generator.state_dict(),
            path + "/bigan_generator.pth")

        torch.save(
            self.bigan.encoder.state_dict(),
            path + "/bigan_encoder.pth")

        torch.save(
            self.bigan.discriminator.state_dict(),
            path + "/bigan_discriminator.pth")

    def loss_function(self, reconstructed_x, x, mu, logvar, use_bce=True):
        """
        Reconstruction + KL divergence losses summed over all elements and
        batch.

        If for the reconstruction loss, the binary cross entropy is used, be
        sure to have an adequate activation for the last layer of the decoder
        (eg a sigmoid). Same goes if binary cross entropy is not used (in that
        case, mean squared error is used, you could use a tanh activation)
        """
        if use_bce:
            reconstruction_loss = F.binary_cross_entropy(
                self.model.flatten(reconstructed_x),
                self.model.flatten(x),
                reduction='sum')
        else:
            reconstruction_loss = F.mse_loss(
                self.model.flatten(reconstructed_x),
                self.model.flatten(x),
                reduction='sum')

        # Adding a beta value for a beta VAE. With beta = 1, standard VAE
        beta = 1.0

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta

        return reconstruction_loss + KLD

    def train(self, epochs, log_interval=10, use_bce=True):
        if self.train_loader is None:
            return

        train_loss = 0
        progress_bar = tqdm(
            total=epochs*len(self.train_loader),
            desc="VAE training",
            leave=False)

        for epoch in range(epochs):
            self.model.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(
                    recon_batch,
                    data,
                    mu,
                    logvar,
                    use_bce=use_bce)

                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                if batch_idx % log_interval == 0:
                    self.writer.add_scalar(
                        'loss/training_loss',
                        train_loss / log_interval,
                        epoch * len(self.train_loader) + batch_idx)

                    train_loss = 0

                progress_bar.update(1)

            self.epoch_test(epoch)

        self.writer.close()

    def epoch_test(self, epoch, use_bce=True):
        if self.test_loader is None:
            return

        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(
                    recon_batch,
                    data,
                    mu,
                    logvar,
                    use_bce=use_bce).item()

                if i == 0:
                    n = min(data.size(0), 8)
                    recon_batch = self.model.unflatten(recon_batch)
                    comparison = torch.cat([data[:n], recon_batch[:n]])

                    if not os.path.exists("results"):
                        os.mkdir("results")

                    filename = 'results/reconstruction_' + str(epoch) + '.png'
                    save_image(comparison.cpu(), filename, nrow=n)

        test_loss /= len(self.test_loader.dataset)
        self.writer.add_scalar(
            'loss/test_loss',
            test_loss / len(self.test_loader.dataset),
            epoch)

    def plot_latent_slice(self, dark_background=False):
        """
        Plots a slice of the latent space manifold (decoded images). The first
        2 dimensions of the latent space are sampled uniformily, the remaining
        are set to 0
        """
        self.model.eval()
        filename = os.path.join("results/digits_over_latent.png")

        # display a 30x30 2D manifold of images (we consider that the images
        # have the same width and height)
        n = 30
        image_size = self.model.width

        figure = np.zeros((
            self.model.channels,
            image_size * n,
            image_size * n))

        if self.model.channels > 1:
            shape = (self.model.channels, image_size, image_size)
        else:
            shape = (image_size, image_size)

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-10.0, 10.0, n)
        grid_y = np.linspace(-10.0, 10.0, n)[::-1]

        with torch.no_grad():
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = np.zeros((1, self.model.z_dim))
                    z_sample[0][0] = xi
                    z_sample[0][1] = yi
                    z_tensor = torch.from_numpy(z_sample).float().to(
                        self.device)

                    z_tensor = self.model.fc3(z_tensor)
                    x_decoded = self.model.decoder(z_tensor)
                    image = x_decoded.cpu().detach().numpy().reshape(shape)

                    figure[
                        :,
                        i * image_size: (i + 1) * image_size,
                        j * image_size: (j + 1) * image_size] = image

        figure = figure.transpose(1, 2, 0)

        if self.model.channels == 1:
            figure = figure[:, :, 0]

        if dark_background:
            plt.style.use('dark_background')

        plt.figure(figsize=(10, 10))
        start_range = image_size // 2
        end_range = (n - 1) * image_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, image_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.show()

    def plot_results(self, dark_background=False):
        """
        Plots labels as a function of the 2D latent vector
        """
        if self.test_loader is None:
            return

        z_mean = np.zeros((
            len(self.test_loader.dataset),
            self.model.getZDim()))

        targets = np.zeros(len(self.test_loader.dataset))
        idx = 0

        for data, target in self.test_loader:
            data = data.to(self.device)

            if isinstance(self.model, FCVAE):
                mu, logvar = self.model.encode(self.model.flatten(data))
                z = self.model.reparameterize(mu, logvar)
            else:
                z = self.model.representation(data)

            np_z = z.cpu().detach().numpy()
            np_target = target.detach().numpy()
            batch_size = np_z.shape[0]

            z_mean[idx:idx + batch_size] = np_z
            targets[idx:idx + batch_size] = np_target

            idx += batch_size

        if dark_background:
            plt.style.use('dark_background')

        fig = plt.figure(figsize=(12, 10))

        if self.model.getZDim() == 2:
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=targets)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.colorbar()

        elif self.model.getZDim() == 3:
            ax = fig.add_subplot(111, projection='3d')
            cloud = ax.scatter(
                z_mean[:, 0],
                z_mean[:, 1],
                z_mean[:, 2],
                c=targets)

            ax.set_xlabel("z[0]")
            ax.set_ylabel("z[1]")
            ax.set_zlabel("z[2]")
            fig.colorbar(cloud)

        else:
            print("Latent space dimension should be 2 or 3 to be displayed")
            return

        plt.title(
            "Latent space of the VAE")

        if not os.path.exists("results"):
            os.mkdir("results")

        plt.savefig("results/vae.png")
        plt.show()
