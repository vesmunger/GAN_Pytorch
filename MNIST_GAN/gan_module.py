import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from models.generator import Generator
from models.discriminator import Discriminator

class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        # Fixed noise for visualization
        self.validation_z = torch.randn(6, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        opt_g, opt_d = self.optimizers()

        # Train generator
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim, device=self.device)
        fake_imgs = self(z)
        g_loss = self.adversarial_loss(self.discriminator(fake_imgs), torch.ones(real_imgs.size(0), 1, device=self.device))
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # Train discriminator
        y_real = torch.ones(real_imgs.size(0), 1, device=self.device)
        y_fake = torch.zeros(real_imgs.size(0), 1, device=self.device)

        real_loss = self.adversarial_loss(self.discriminator(real_imgs), y_real)
        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), y_fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_g, opt_d], []

    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z).cpu()
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i + 1)
            plt.imshow(sample_imgs[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.axis('off')
        plt.show()