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

try:
    from sklearn.manifold import TSNE

except ImportError:
    pass


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
            lr=self.lr,
            betas=[self.beta_1, self.beta_2],
            weight_decay=self.weight_decay)

        self.optimizer_d = optim.Adam(
            self.bigan.discriminator.parameters(),
            lr=self.lr,
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

    def load_net(self, path):
        """
        Loads a bigan. The manager will look for the files bigan_generator.pth,
        bigan_encoder.pth and bigan_discriminator.pth at the specified path
        """
        generator_state_dict = torch.load(path + "/bigan_generator.pth")
        encoder_state_dict = torch.load(path + "/bigan_encoder.pth")
        discriminator_state_dict = torch.load(
            path + "/bigan_discriminator.pth")

        self.bigan.generator.load_state_dict(generator_state_dict)
        self.bigan.encoder.load_state_dict(encoder_state_dict)
        self.bigan.discriminator.load_state_dict(discriminator_state_dict)

    def save_net(self, path):
        """
        Saves the bigan currently loaded in the manager, at the specified path
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

    def compute_losses(self, discriminator_gen, discriminator_enc):
        """
        Computes the losses for the BiGAN network.

        Parameters:
            discriminator_gen - Discriminator(Generator(z), z)
            discriminator_enc - Discriminator(x, Encoder(x))

        Returns:
            loss_g - The generator loss
            loss_d - The discriminator loss
        """
        noise = 1e-8
        loss_g = -torch.mean(
            torch.log(discriminator_gen + noise) +
            torch.log(1 - discriminator_enc + noise))

        loss_d = -torch.mean(
            torch.log(discriminator_enc + noise) +
            torch.log(1 - discriminator_gen + noise))

        return loss_g, loss_d

    def train(self, epochs, log_interval=10):
        if self.train_loader is None:
            return

        train_loss_g = 0
        train_loss_d = 0
        progress_bar = tqdm(
            total=epochs*len(self.train_loader),
            desc="BiGAN training",
            leave=False)

        for epoch in range(epochs):
            self.bigan.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                z = torch.rand(
                    data.size(0),
                    self.bigan.generator.get_z_dim()).to(self.device)

                x = data.to(self.device)

                x_hat = self.bigan.generator(z)
                z_hat = self.bigan.encoder(x)

                discriminator_enc = self.bigan.discriminator(x, z_hat)
                discriminator_gen = self.bigan.discriminator(x_hat, z)

                loss_g, loss_d = self.compute_losses(
                    discriminator_gen,
                    discriminator_enc)

                loss_g.backward(retain_graph=True)
                self.optimizer_g.step()
                self._reset_grads()

                loss_d.backward(retain_graph=True)
                self.optimizer_d.step()
                self._reset_grads()

                # Potiential bug source, to be checked
                train_loss_g += loss_g.item()
                train_loss_d += loss_d.item()

                if batch_idx % log_interval == 0:
                    self.writer.add_scalar(
                        'loss/training_loss_generator',
                        train_loss_g / log_interval,
                        epoch * len(self.train_loader) + batch_idx)

                    self.writer.add_scalar(
                        'loss/training_loss_discriminator',
                        train_loss_d / log_interval,
                        epoch * len(self.train_loader) + batch_idx)

                    train_loss_g = 0
                    train_loss_d = 0

                progress_bar.update(1)

            self.epoch_test(epoch)

        self.writer.close()

    def epoch_test(self, epoch):
        if self.test_loader is None:
            return

        self.bigan.eval()
        test_loss_g = 0
        test_loss_d = 0
        mean_pixel_norm = 0
        mean_z_norm = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                z = torch.rand(
                    data.size(0),
                    self.bigan.generator.get_z_dim()).to(self.device)

                x = data.to(self.device)

                x_hat = self.bigan.generator(z)
                z_hat = self.bigan.encoder(x)

                discriminator_enc = self.bigan.discriminator(x, z_hat)
                discriminator_gen = self.bigan.discriminator(x_hat, z)

                loss_g, loss_d = self.compute_losses(
                    discriminator_gen,
                    discriminator_enc)

                test_loss_g += loss_g.item()
                test_loss_d += loss_d.item()

                # Compute G(E(x)) and E(G(z))
                generated_z_hat = self.bigan.generator(z_hat)
                encoded_x_hat = self.bigan.encoder(x_hat)

                # Evaluate the pixel norm, or the {x, G(E(x))} gap
                pixel_norm = x - generated_z_hat
                pixel_norm = pixel_norm.norm().item()
                mean_pixel_norm += pixel_norm

                # Evaluate the z norm, or the {z, E(G(z)} gap
                z_norm = z - encoded_x_hat
                z_norm = z_norm.norm().item()
                mean_z_norm += z_norm

                if i == 0:
                    n = min(x.size(0), 8)
                    comparison = torch.cat([x[:n], generated_z_hat[:n]])

                    if not os.path.exists("results"):
                        os.mkdir("results")

                    filename = 'results/reconstruction_' + str(epoch) + '.png'
                    save_image(comparison.cpu(), filename, nrow=n)

        test_loss_g /= len(self.test_loader.dataset)
        test_loss_d /= len(self.test_loader.dataset)
        mean_pixel_norm /= len(self.test_loader.dataset)
        mean_z_norm /= len(self.test_loader.dataset)

        self.writer.add_scalar(
            'loss/test_loss_g',
            test_loss_g,
            epoch)

        self.writer.add_scalar(
            'loss/test_loss_d',
            test_loss_d,
            epoch)

        self.writer.add_scalar(
            'norm/pixel_norm',
            mean_pixel_norm,
            epoch)

        self.writer.add_scalar(
            'norm/z_norm',
            mean_z_norm,
            epoch)

    def plot_latent_slice(self, dark_background=False):
        """
        Plots a slice of the latent space manifold (decoded images). The first
        2 dimensions of the latent space are sampled uniformily, the remaining
        are set to 0
        """
        self.bigan.eval()
        filename = os.path.join("results/latent_slice.png")

        # display a 30x30 2D manifold of images (we consider that the images
        # have the same width and height)
        n = 30
        image_size = self.bigan.generator.get_width()

        figure = np.zeros((
            self.bigan.generator.get_nb_channels(),
            image_size * n,
            image_size * n))

        if self.bigan.generator.get_nb_channels() > 1:
            shape = (
                self.bigan.generator.get_nb_channels(),
                image_size,
                image_size)
        else:
            shape = (image_size, image_size)

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(0.0, 1.0, n)
        grid_y = np.linspace(0.0, 1.0, n)[::-1]

        with torch.no_grad():
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = np.zeros((1, self.bigan.generator.get_z_dim()))
                    z_sample[0][0] = xi
                    z_sample[0][1] = yi
                    z_tensor = torch.from_numpy(z_sample).float().to(
                        self.device)

                    # z_tensor = torch.rand(
                    #     1,
                    #     self.bigan.generator.get_z_dim()).to(self.device)

                    x_hat = self.bigan.generator(z_tensor)
                    image = x_hat.cpu().detach().numpy().reshape(shape)

                    figure[
                        :,
                        i * image_size: (i + 1) * image_size,
                        j * image_size: (j + 1) * image_size] = image
        return
        figure = figure.transpose(1, 2, 0)

        if self.bigan.generator.get_nb_channels() == 1:
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
        plt.title("Generated images for a slice of the BiGAN latent space")
        plt.savefig(filename)
        plt.show()

    def plot_latent_space(self, dark_background=False, dimensions=2):
        """
        Plots the images of the test set projected onto the latent space
        (through the encoder of the BiGAN), with their respective labels for
        more clarity.

        Note that if the dimension of the latent space is > 2, T-SNE is used
        for the visualization (reducing the dimension of the data)
        """
        if self.test_loader is None:
            return

        z_dim = self.bigan.encoder.get_z_dim()
        z_array = np.zeros((
            len(self.test_loader.dataset),
            z_dim))

        targets = np.zeros(len(self.test_loader.dataset))
        idx = 0

        for data, target in self.test_loader:
            x = data.to(self.device)
            z_hat = self.bigan.encoder(x)
            np_z = z_hat.cpu().detach().numpy()
            np_target = target.detach().numpy()
            batch_size = np_z.shape[0]

            z_array[idx:idx + batch_size] = np_z[:, :]
            targets[idx:idx + batch_size] = np_target

            idx += batch_size

        if dark_background:
            plt.style.use('dark_background')

        fig = plt.figure(figsize=(12, 10))

        if z_dim > dimensions:
            z_array = TSNE(n_components=dimensions).fit_transform(z_array)

        if z_array.shape[1] == 2:
            plt.scatter(z_array[:, 0], z_array[:, 1], c=targets)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.colorbar()

        elif z_array.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
            cloud = ax.scatter(
                z_array[:, 0],
                z_array[:, 1],
                z_array[:, 2],
                c=targets)

            ax.set_xlabel("z[0]")
            ax.set_ylabel("z[1]")
            ax.set_zlabel("z[2]")
            fig.colorbar(cloud)

        else:
            print("The dimensions param should be set to 2 or 3")
            return
        
        if z_dim > dimensions:
            title = "T-SNE visualization of the BiGAN latent space. "
            title += "Original dimension: " + str(z_dim)
        else:
            title = "Latent space of the BiGAN"

        plt.title(title)

        if not os.path.exists("results"):
            os.mkdir("results")

        plt.savefig("results/latent_space.png")
        plt.show()

    def _reset_grads(self):
        """
        Calls zero_grad on all the networks of the BiGAN
        """
        self.bigan.generator.zero_grad()
        self.bigan.encoder.zero_grad()
        self.bigan.discriminator.zero_grad()
