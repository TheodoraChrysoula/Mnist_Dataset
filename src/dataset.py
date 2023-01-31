#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import medmnist 
from medmnist import INFO, Evaluator


# In[3]:


print(f'MEDMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}')


# In[4]:


data_flag = 'pneumoniamnist'
#data_flag = 'organamnist'
# data_flag = 'breastmnist'
download = 'True'

epochs = 50
BATCH_SIZE = 32
lr = 0.001
num_workers = 2 # number of parallel processes for data preparation

info = INFO[data_flag]
task = info['task'] # multiclass
n_channels = info['n_channels'] # 1 channel
n_classes = len(info['label']) # 11 classes
class_names = info['label']
DataClass = getattr(medmnist, info['python_class'])


# In[11]:


print (f"The number of classes is: {n_classes} \n The number of channels is {n_channels}.")


# In[14]:


print(f"The class_names are {class_names}")


# In[15]:


print(info)


# In[32]:


data_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5,9),sigma = (0.1, 5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[.5])
])


# Load the data
train_dataset = DataClass(split ='train', transform = data_transform, download=download)
test_dataset = DataClass(split='test', transform = data_transform, download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = num_workers)
train_loader_at_eval = data.DataLoader(dataset = train_dataset, batch_size = 2*BATCH_SIZE, shuffle = False, num_workers = num_workers)
test_loader = data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False, num_workers = num_workers)


# In[37]:


print(train_dataset)
print('='*50)
print(test_dataset)


# ### Visualization
# 
# 

# In[47]:


train_dataset.montage(length=1)


# In[48]:


train_dataset.montage(length=20)


# In[157]:


import matplotlib
samples = np.random.choice(1000, 3)
print(samples)
I = len(samples)
matplotlib.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(1, 3, figsize=(6, 2*I))


for i in range(I):
    img = train_dataset[samples[i]][0][0]
    lab = train_dataset[samples[i]][1][0]
    ax[i].imshow(img, cmap='gray')
    ax[i].set_title(class_names[f"{lab}"])
    ax[i].xaxis.set_major_locator(plt.NullLocator())
    ax[i].yaxis.set_major_locator(plt.NullLocator())
    


# In[ ]:




