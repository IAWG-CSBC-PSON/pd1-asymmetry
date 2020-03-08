import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

# from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import umap
import pandas as pd
import numpy as np
import json
import datetime
import os
from . import vae


def Cloud2Grid(coords, grid_dim=10000, tile_size=256):
    """ Convert points into a grid
    Arguments
        coords(DataFrame): 2-column frame of x- and y-coordinates
        grid_dim (int): output dimension of the final image
        tile_size (int): size of the tiles to paste
    Returns
        coords(DataFrame)
    """
    grid_coords = np.array(coords)
    tile_dim = grid_dim // tile_size  # tile count in x and y
    for i in range(2):  # scale axes from 0 to tile_dim
        grid_coords[:, i] = grid_coords[:, i] - grid_coords[:, i].min()
        grid_coords[:, i] = grid_coords[:, i] * (
            tile_dim / (grid_coords[:, i].max() + 1)
        )
    grid_coords = np.floor(grid_coords)
    grid_coords = grid_coords * tile_size
    grid_coords[:, 1] = abs(grid_coords[:, 1] - grid_dim)  # flip y axis index
    return grid_coords


def MakeCoordPlot(tiles, coords, image_size=10000, boarder_width=20):
    """ Plot individual images as tiles according to provided coordinates
    Arguments
        data_file (str):
        annot_file (str)
        image_size (int)
    Returns
        grid image projection and updated coordinates
    """
    tile_size = tiles.shape[1]

    grid_coords = Cloud2Grid(
        coords, grid_dim=(image_size - 2 * tile_size), tile_size=tile_size
    )
    grid_coords = grid_coords + tile_size  # for black boarder
    grid_image = Image.new("RGB", (image_size, image_size))
    for i in range(len(tiles)):  # paste each tile onto image
        tile = ColorTileBoarder(tiles[i], channel=0, boarder_width=2)
        tile = Image.fromarray(tiles[i])
        x, y = grid_coords[i, :]
        grid_image.paste(tile, (int(x), int(y)))
    coords["grid1"] = grid_coords[:, 0] + tile_size // 2
    coords["grid2"] = grid_coords[:, 1] + tile_size // 2
    return grid_image, coords


def ColorTileBoarder(tile, channel, boarder_width=20):
    """ Generate a colored boarder for the tile
    """
    assert channel < 4  # only colored images for now
    fill = np.zeros(3).astype(np.uint8)
    if channel != 0:
        fill[int(channel) - 1] = 0
    mask = np.ones(tile.shape)
    mask[boarder_width:-boarder_width, boarder_width:-boarder_width, ...] = 0
    np.place(tile, mask, vals=fill)
    return tile


def show_image(d):
    # d = np.moveaxis(d.numpy(), 0, 2)
    d = d.detach().numpy()[0, ...]
    d = (255 * d).astype(np.uint8)
    d = np.dstack((d, d, d))
    di = Image.fromarray(d)
    di.show()


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(
        recon_x.view(-1, x.shape[0]), x.reshape(-1, x.shape[0]), size_average=False
    )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + (0.2 * KLD)


def train_vae(
    all_tiles, outf, nz=8, batch_size=16, epochs=20, cuda=True, seed=1, log_interval=10
):
    cuda = cuda and torch.cuda.is_available()

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    print("Use CUDA:", cuda, "CUDA available:", torch.cuda.is_available())

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    vae_data = vae.VAEDataset(all_tiles)

    train_loader = DataLoader(vae_data, batch_size=4, shuffle=True, num_workers=4)

    test_loader = DataLoader(vae_data, batch_size=4, shuffle=False, num_workers=4)

    model = vae.VAE(image_channels=1, z_dim=nz)
    model.have_cuda = cuda

    if cuda:
        model.cuda()


    print(model)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    outf = outf + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if not os.path.exists(outf):
        os.makedirs(outf)
    outf_samples = outf + "/samples"
    os.makedirs(outf_samples)

    with open(outf + "/model.txt", "w") as f1:
        print(model, file=f1)

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = Variable(data)
            if cuda:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, z, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item() / len(data),
                    )
                )
        print(
            "====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "%s/vae_epoch_%d.pth" % (outf, epoch))

    def encode():
        model.eval()
        encodings = pd.DataFrame()
        for batch_idx, data in enumerate(test_loader):
            data = Variable(data)
            if cuda:
                data = data.cuda()
            recon_batch, z, mu, logvar = model(data)
            encodings = encodings.append(pd.DataFrame(z.detach().numpy()))
        return encodings

    for epoch in range(1, epochs + 1):
        train(epoch)

    encodings = encode()

    umap_embed = umap.UMAP().fit_transform(encodings)

    # make coord plot
    all_tiles = np.stack([i / i.max() for i in all_tiles])
    img, _ = MakeCoordPlot(
        ((all_tiles * 2 ** 8).astype(np.uint8)),
        pd.DataFrame(umap_embed),
        image_size=2000,
    )

    file_prefix = (
        "batchsize_"
        + str(batch_size)
        + "-"
        + "epochs_"
        + str(epochs)
        + "-"
        + "latentdim_"
        + str(nz)
    )

    encodings.to_csv(file_prefix + "_encodings.csv", sep=",", index=False, header=False)
    pd.DataFrame(umap_embed).to_csv(
        file_prefix + "umap_encodings.csv", sep=",", header=False, index=False
    )
    img.save(file_prefix + "umap_projection.png")
