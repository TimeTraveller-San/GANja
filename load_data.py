import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os

import matplotlib.pyplot as plt

class DataGenerator(Dataset):
    def __init__(self, directory, transform=None, n_samples=np.inf):
        self.directory = directory
        self.transform = transform
        self.n_samples = n_samples
        self.samples = self.load_dogs_data(directory)

    def load_dogs_data(self, directory, imgsize=64):
        required_transforms = transforms.Compose([
                transforms.Resize(imgsize),
                transforms.CenterCrop(imgsize),
        ])
        imgs = []
        paths = []
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in sorted(fnames)[:min(self.n_samples, 999999999999999)]:
                path = os.path.join(root, fname)
                paths.append(path)

        for path in paths:
            # Load image
            try:
                img = dset.folder.default_loader(path)
            except:
                continue
            object_img = required_transforms(img)
            imgs.append(object_img)
        return imgs

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.transform is not None:
            sample = self.transform(sample)
        return np.asarray(sample)

    def __len__(self):
        return len(self.samples)

def show_img(imgs, n=5, denorm=True):
    sample = []
    for i in range(n):
        img = imgs[i].to("cpu").clone().detach().squeeze(0)
        img = img.numpy().transpose(1, 2, 0)
        sample.append(img)

    figure, axes = plt.subplots(1, len(sample), figsize=(64, 64))
    for i, axis in enumerate(axes):
        axis.axis('off')
        img_array = sample[i]
        if denorm: axis.imshow(img_array*0.5 + 0.5) #De-normalize
        else: axis.imshow(img_array)
