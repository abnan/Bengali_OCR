import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import os
from skimage import io, transform

class BengaliCharactersDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.X, self.y = data[0], data[1]
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.X.iloc[idx, 0])
        image = io.imread(img_name+".jpg")
        if self.transform:
            image = self.transform(image)
        components = self.y.iloc[idx]
        components = torch.tensor([components])
        sample = {'image': image, 'components': components}

        return sample