import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
from skimage import io, transform
from utils import BengaliCharactersDataset
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


df_train_csv = pd.read_csv("train.csv")

X = df_train_csv.pop('image_id').to_frame()
df_train_csv.pop('grapheme')
y = df_train_csv


from sklearn.model_selection import train_test_split


# Create 20% validation dataset
X_train, X_val, y_train, y_val = train_test_split(
        X, y,stratify=y, test_size=0.2)

data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.9469, 0.9469, 0.9469],
                             std=[0.1623, 0.1623, 0.1623]),
    ])

train_dataset = BengaliCharactersDataset(data=(X_train, y_train), root_dir='data/entire_dataset/', transform = data_transform)
val_dataset = BengaliCharactersDataset(data=(X_val, y_val), root_dir='data/entire_dataset/', transform = data_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=10)

def calc_dataset_loss(data_loader):
    total = 0

    correct_grapheme = 0

    correct_vowel = 0

    correct_consonant = 0

    for i in data_loader:
        CustomNet.eval()
        optimizer.zero_grad()
        test_out = CustomNet(i['image'].to(device))

        predicted_grapheme  = torch.argmax(test_out[0], dim=1)
        labels_grapheme = i['components'][:,:,0].squeeze().to(device)
        correct_grapheme += (predicted_grapheme  == labels_grapheme).sum().item()

        predicted_vowel = torch.argmax(test_out[1], dim=1)
        labels_vowel = i['components'][:,:,1].squeeze().to(device)
        correct_vowel += (predicted_vowel == labels_vowel).sum().item()

        predicted_consonant = torch.argmax(test_out[2], dim=1)
        labels_consonant = i['components'][:,:,2].squeeze().to(device)
        correct_consonant += (predicted_consonant == labels_consonant).sum().item()
        
        total += labels_consonant.shape[0]


    return (correct_grapheme/total*100, correct_vowel/total*100, correct_consonant/total*100)


class AppendNet(nn.Module):
    def __init__(self):
        super(AppendNet, self).__init__()
        self.start = nn.Linear(1000, 1000)
        self.fc_root = nn.Linear(1000, 168)
        self.fc_vowel = nn.Linear(1000, 11)
        self.fc_consonant = nn.Linear(1000, 7)
    
    def forward(self, x):
        root_prob = self.fc_root(F.relu(self.start(x)))
        vowel_prob = self.fc_vowel(F.relu(self.start(x)))
        consonant_prob = self.fc_consonant(F.relu(self.start(x)))
        return root_prob, vowel_prob, consonant_prob

vgg = models.vgg11(True)
# for param in vgg.parameters():
#     param.requires_grad = False

CustomNet = nn.Sequential(vgg, AppendNet()).to(device)

criterion = nn.CrossEntropyLoss()


loss1_list = []
loss2_list = []
loss3_list = []
cnt=0

params_to_update = []
for name,param in CustomNet.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)


writer1 = SummaryWriter('runs/grapheme_root_log')
writer2 = SummaryWriter('runs/vowel_diacritic_log')
writer3 = SummaryWriter('runs/consonant_diacritic_log')

optimizer = optim.Adam(params_to_update, lr=5e-4, amsgrad=True)

print("Training accuracy:")
train_stats = calc_dataset_loss(train_dataloader)
print("Grapheme:", train_stats[0], "Vowel:", train_stats[1], "Consonant:", train_stats[2])
print("Validation accuracy:")
val_stats = calc_dataset_loss(val_dataloader)
print("Grapheme:", val_stats[0], "Vowel:", val_stats[1], "Consonant:", val_stats[2])

for epoch in range(20):
    running_loss1 = 0
    running_loss2 = 0
    running_loss3 = 0
    for i in train_dataloader:
        CustomNet.train()
        cnt+=1
        optimizer.zero_grad()
        test_out = CustomNet(i['image'].to(device))
        loss1 = criterion(test_out[0], i['components'][:,:,0].squeeze(1).to(device))
        loss2 = criterion(test_out[1], i['components'][:,:,1].squeeze(1).to(device))
        loss3 = criterion(test_out[2], i['components'][:,:,2].squeeze(1).to(device))

        loss1_list.append(loss1.item())
        loss2_list.append(loss2.item())
        loss3_list.append(loss3.item())

        running_loss1+=loss1.item()
        running_loss2+=loss2.item()
        running_loss3+=loss3.item()

        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()

        optimizer.step()
    print(epoch)
    print("Training accuracy:")
    train_stats = calc_dataset_loss(train_dataloader)
    print("Grapheme:", train_stats[0], "Vowel:", train_stats[1], "Consonant:", train_stats[2])
    print("Validation accuracy:")
    val_stats = calc_dataset_loss(val_dataloader)
    print("Grapheme:", val_stats[0], "Vowel:", val_stats[1], "Consonant:", val_stats[2])
    writer1.add_scalar('loss', running_loss1/len(train_dataloader), epoch)
    writer2.add_scalar('loss', running_loss2/len(train_dataloader), epoch)
    writer3.add_scalar('loss', running_loss3/len(train_dataloader), epoch)
    writer1.flush()
    writer2.flush()
    writer3.flush()
writer1.close()
writer2.close()
writer3.close()
