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
from utils import BengaliCharactersDataset



df_train_0 = pd.read_parquet("train_image_data_0.parquet")
df_train_1 = pd.read_parquet("train_image_data_1.parquet")
df_train_2 = pd.read_parquet("train_image_data_2.parquet")
df_train_3 = pd.read_parquet("train_image_data_3.parquet")

df_train = pd.concat([df_train_0, df_train_1, df_train_2, df_train_3], axis=0)

h = 137
w = 236

# for i in range(len(df_train)):
#     im = Image.fromarray(df_train.iloc[i, 1:].values.reshape(h,w).astype(np.uint8))
#     im.save("data/entire_dataset/" + df_train.iloc[i, 0]+".jpg")


data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

df_train_csv = pd.read_csv("train.csv")
X = df_train_csv.pop('image_id').to_frame()
df_train_csv.pop('grapheme')
y = df_train_csv

train_dataset = BengaliCharactersDataset(data=(X, y), root_dir='data/entire_dataset/', transform = data_transform)

loader = DataLoader(
    train_dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)


mean = 0.
std = 0.
nb_samples = 0.
cnt=0
for merged_data in loader:
    cnt+=1
    data = merged_data['image']
#     print(batch_samples.shape)
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
    if cnt%1000==0:
        print(cnt)
        # break

mean /= nb_samples
std /= nb_samples
print("mean = ", mean, "std = ", std)