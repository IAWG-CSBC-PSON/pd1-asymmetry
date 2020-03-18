import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import Dataset
import torch


class VAEDataset(Dataset):
    def __init__(self, data_file, transform=None, do_normalize=True):
        """
        Args:
            data_file (string, np.array): Path to the npy file with images or numpy array.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if not isinstance(data_file, np.ndarray):
            self.data = np.load(data_file)
        else:
            self.data = data_file
        if do_normalize:
            self.normalize()
        self.data = self.data[..., np.newaxis]  # pytorch requirement
        self.data = np.moveaxis(self.data, 3, 1)
        self.data = torch.Tensor(self.data)
        self.transform = transform

    def normalize(self):
        self.data = np.stack([(i - i.min()) / (i.max() - i.min()) for i in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs = self.data[idx, ...]
        if self.transform:
            imgs = self.transform(imgs)
        return imgs


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, *args, **kwargs):
        self.image_size = kwargs.pop("image_size", 1)
        super().__init__(*args, **kwargs)

    def forward(self, input, size=1024):
        s = input.size()
        return input.view(
            s[0], s[1] // (self.image_size ** 2), self.image_size, self.image_size
        )


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.h_dim = h_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(409600, h_dim),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.have_cuda:
            std.cuda()
            esp = esp.cuda()
            mu.cuda()
            logvar.cuda()
        # import ipdb
        # ipdb.set_trace()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        di = self.fc3(z)
        return self.decoder(di), z, mu, logvar
