import matplotlib.pyplot as plt
import configparser
import argparse
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

from load_data import DataGenerator, show_img
from networks import Generator, Discriminator

from utils import show_generated_img, init_dirs
from load_data import DataGenerator, show_img

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_state(filepath, epoch, netG, netD, optimizerG, optimizerD):
    state = {
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
    }
    torch.save(state, filepath)

def load_state(filepath):
    return torch.load(filepath)


# Initialize all params
config = configparser.ConfigParser()
config.read('config.ini')
conf = config['DEFAULT']

N = 25000 # To limit number of images to load for training
batch_size = conf.getint('batch_size')
database = conf['data_dir']
lr_d = conf.getfloat('lr_d')
lr_g = conf.getfloat('lr_g')
beta1 = conf.getfloat('beta1')
beta2 = conf.getfloat('beta1')
epochs = conf.getint('epochs')
nz = conf.getint('nz')
real_label = conf.getfloat('real_label')
fake_label = conf.getfloat('fake_label')
hist_average_cutoff = conf.getint('hist_average_cutoff')
img_save_freq = conf.getint('img_save_freq')
model_save_freq = conf.getint('model_save_freq')
logs_dir = conf['logs_dir']
img_dir, model_dir = init_dirs(logs_dir)

print("Loading data...")
random_transforms = [transforms.RandomRotation(degrees=5)]

transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                # transforms.RandomApply(random_transforms,
                                                    # p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))])

train_data = DataGenerator(database, transform=transform, n_samples=N)

train_loader = DataLoader(train_data, shuffle=True,
                                           batch_size=batch_size, num_workers=4)



netG = Generator(128, 32, 3).to(device)
netD = Discriminator(3, 48).to(device)

criterion = nn.BCELoss()
criterionH = nn.MSELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                optimizerG,
                                                T_0=epochs//200,
                                                eta_min=0.00005)
lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                optimizerD,
                                                T_0=epochs//200,
                                                eta_min=0.00005)

batch_size = train_loader.batch_size

def get_model_weights(net):
    average = {}
    params = dict(net.named_parameters())
    for p in params:
        average[p] = params[p].detach()
    return average

averageD = False
averageG = False
step = 1

# load_path = False
load_path = "logs/models/1250.torch"
# load_path = False

t = tqdm(total=epochs)
epoch = 0
while epoch <= epochs:
    if load_path:
        checkpoint = load_state(load_path)
        epoch = checkpoint['epoch']
        t.update(epoch)
        print(epoch)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        load_path = False

    epoch += 1
    t.update(1)
    for ii, (real_images) in tqdm(enumerate(train_loader),
                                                    total=len(train_loader)):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()

        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device) + np.random.uniform(-0.1, 0.1)

        output = netD(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # Historical averaging weights
        err_hD = 0
        if epoch > hist_average_cutoff:
            if not averageD:
                print("Starting historical weight averaging for discriminator")
                averageD = get_model_weights(netD)
            paramsD = dict(netD.named_parameters())
            for p in paramsD:
                err_hD += criterionH(paramsD[p], averageD[p])
                averageD[p] = (averageD[p]*(step-1) + paramsD[p].detach())/step
            err_hD.backward()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        labels.fill_(fake_label) + np.random.uniform(0, 0.2)
        output = netD(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD_final = errD_real + errD_fake + err_hD
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()

        err_hG = 0
        if epoch > hist_average_cutoff:
            if not averageG:
                print("Starting historical weight averaging for generator")
                averageG = get_model_weights(netG)
            paramsG = dict(netG.named_parameters())
            for p in paramsG:
                err_hG += criterionH(paramsG[p], averageG[p])
                averageG[p] = (averageG[p]*(step-1) + paramsG[p].detach())/step
            err_hG.backward()
            step += 1

        errG_final = errG + err_hG
        optimizerG.step()
        lr_schedulerG.step(epoch)
        lr_schedulerD.step(epoch)

    if epoch % img_save_freq == 0:
        info = f"""
                    [{epoch}/{epochs}][{ii}/{len(train_loader)}]
                    Loss_D: {errD_final.item():.4f} Loss_G: {errG_final.item():.4f}
                    D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}
            """
        print(info)
        imgName = f"{epoch}.png"
        fname = os.path.join(img_dir, imgName)
        show_generated_img(netG, 10, fname=fname)

    if epoch % model_save_freq == 0:
        modelName = f"{epoch}.torch"
        filepath = os.path.join(model_dir, modelName)
        save_state(filepath, epoch, netG, netD, optimizerG, optimizerD)
t.close()
