import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.ndimage import rotate

# from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import umap
import pandas as pd
import numpy as np
import copy
import json
import datetime
import os
from pathlib import Path
from operator import itemgetter
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
    return BCE + (0.1 * KLD)


def rotate_cell(cell, theta=None):
    if theta is None:
        mr, mc = np.where(cell == np.amax(cell))  # find brightest point
        cr, cc = (cell.shape[0] // 2, cell.shape[1] // 2)  # wrt center
        theta = np.arctan2(mr - cr, mc - cc)[0]  # compute angle wrt center
    ri = rotate(
        cell, angle=theta * (180 / np.pi), reshape=False
    )  # scipy.ndimage.rotate
    return theta, ri


def random_transform(data):
    data = rotate(data, angle=np.random.randint(0, 360), axes=(1, 2), reshape=False)
    if np.random.random() > 0.5:
        # Remove negative stride with copy
        data = np.flip(data, axis=1).copy()
    return data


def encode(model, data_loader):
    model.eval()
    encodings = pd.DataFrame()
    for batch_idx, data in enumerate(data_loader):
        data = Variable(data)
        if getattr(model, "have_cuda", False):
            data = data.cuda()
        recon_batch, z, mu, logvar = model(data)
        encodings = encodings.append(pd.DataFrame(z.cpu().detach().numpy()))
    return encodings


def decode(model, data_loader):
    model.eval()
    decodings = []
    for batch_idx, data in enumerate(data_loader):
        data = Variable(data)
        if getattr(model, "have_cuda", False):
            data = data.cuda()
        dec = model.decoder(model.fc3(data))
        decodings.append(dec.cpu().detach().numpy())
    return np.concatenate(decodings)


def train_vae(
    all_tiles,
    outf,
    image_channel=0,
    nz=8,
    cpus=2,
    batch_size=16,
    epochs=20,
    cuda=True,
    seed=1,
    log_interval=10,
    learning_rate=5e-4,
    rotate=True,
    normalize=True,
    augment=True,
    model_instance=None,
):
    params = {
        k: v
        for k, v in sorted(locals().items(), key=itemgetter(0))
        if k not in ["all_tiles", "outf", "model_instance"]
    }
    cuda = cuda and torch.cuda.is_available()

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    print("Use CUDA:", cuda, "; CUDA available:", torch.cuda.is_available())

    if image_channel is not None:
        all_tiles = all_tiles[:, image_channel, :, :]
    if rotate:
        all_tiles = np.stack([rotate_cell(i)[1] for i in all_tiles])

    train_dataset = vae.VAEDataset(
        all_tiles,
        do_normalize=normalize,
        transform=random_transform if augment else None,
    )
    test_dataset = copy.copy(train_dataset)
    test_dataset.transform = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpus
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=cpus
    )

    model = (
        vae.VAE(image_channels=1, z_dim=nz)
        if model_instance is None
        else model_instance
    )
    model.have_cuda = cuda

    if cuda:
        model.cuda()

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    outf = Path(outf + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not outf.exists():
        outf.mkdir()

    with open(outf / "model.txt", "w") as f1:
        print(model, file=f1)

    with open(outf / "params.txt", "w") as f1:
        print("\n".join(f"{k}: {v}" for k, v in params.items()), file=f1)

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

    for epoch in range(1, epochs + 1):
        try:
            train(epoch)
        except KeyboardInterrupt:
            print("Interrupted, saving model state...")
            torch.save(model, str(outf / f"trained_model_epoch_{epoch}.pkl"))
            encodings = encode(model, test_loader)
            encodings.to_csv(
                outf / f"encodings_epoch_{epoch}.csv",
                sep=",",
                index=False,
                header=False,
            )
            raise

    encodings = encode(model, test_loader)

    umap_embed = umap.UMAP().fit_transform(encodings)

    # make coord plot
    all_tiles = np.stack([i / i.max() for i in all_tiles])
    img, _ = MakeCoordPlot(
        (all_tiles * ((2 ** 8) - 1)).astype(np.uint8),
        pd.DataFrame(umap_embed),
        image_size=2000,
        boarder_width=3,
    )

    torch.save(model, str(outf / "trained_model.pkl"))
    encodings.to_csv(outf / "encodings.csv", index=False, header=False)
    pd.DataFrame(umap_embed).to_csv(
        outf / "umap_encodings.csv", header=False, index=False
    )
    img.save(outf / "umap_projection.png")
