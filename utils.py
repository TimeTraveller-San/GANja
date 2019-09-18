import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import os
from time import time
from PIL import Image
import torch.utils.data
import torchvision.datasets as dset
import random
from tqdm import tqdm
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_dirs(dir):
    img_dir = os.path.join(dir, 'images')
    model_dir = os.path.join(dir, 'models')
    if not os.path.exists(dir):
        print("Making directories...")
        os.makedirs(dir)
        os.makedirs(img_dir)
        os.makedirs(model_dir)
    return img_dir, model_dir


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

def show_generated_img_matrix(netG, n_images=(4, 4), nz=128, fname=False, fsize=(64, 64)):
    sample = []
    for _ in range(n_images[0] * n_images[1]):
        noise = torch.randn(1, nz, 1, 1, device=device)
        gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        sample.append(gen_image)

    figure, axes = plt.subplots(n_images[0], n_images[1], figsize=fsize)
    for rindex, row in enumerate(axes):
        for cindex, axis in enumerate(row):
            index = rindex * n_images[1] + cindex
            axis.axis('off')
            image_array = (sample[index] + 1.) / 2.
            axis.imshow(image_array)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
        plt.cla()
        plt.close()
    else: plt.show()

def show_generated_img(netG, n_images=5, nz=128, fname=False):
    sample = []
    for _ in range(n_images):
        noise = torch.randn(1, nz, 1, 1, device=device)
        gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        sample.append(gen_image)

    figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))
    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = (sample[index] + 1.) / 2.
        axis.imshow(image_array)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
        plt.cla()
        plt.close()
    else: plt.show()


class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()
    def forward(self, x, alpha=1e-8):
        batch_size, _, height, width = x.shape
        y = x - x.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size,1, height, width)
        y = torch.cat([x, y], 1)
        return y


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def sndeconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))

class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
