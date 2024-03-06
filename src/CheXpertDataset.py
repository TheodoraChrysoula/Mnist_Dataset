#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import typing
from PIL import Image
import pandas as pd





# In[4]:


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# In[5]:


class MyDatasetBase:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        
    def get_datasets(self):
        raise NotImplementedError
        
    def get_data_laoders(self, batch_size):
        raise NotImplementedError


# In[6]:


class CheXpertDataset(Dataset):
    def __init__(self, dataset_dir, idx, policy, target_shape, split):
        assert split in ['train', 'valid', 'test']
        self.parent_dir = dataset_dir
        self.idx = idx
        self.split = split
        self.target_shape = target_shape
        self.policy = policy
        self.data = pd.read_csv(os.path.join(self.parent_dir, self.split+'_frt.csv'))
        labels = self.data.loc[:, idx].values.astype('int')
        if split=='test':
            image_names = self.data.iloc[:, 0].apply(str).apply(lambda x: os.path.join(self.parent_dir, x))
        else:
            image_names = self.data.iloc[:, 0].apply(str).apply(lambda x: os.path.join(os.path.dirname(self.parent_dir),x))
        
        for i in range(labels.shape[0]):
            for l in range(labels.shape[1]):
                a = labels[i][l]
                if a==1:
                    labels[i][l]=1
                elif a == -1:
                    if self.policy=='diff':
                        if l==1 or l==3 or l==4: # Atelectasis, Edema, Pleural Effusion
                            labels[i][l]=1
                        elif l==0 or l==2: # Consolidation, Cardiomegaly
                            labels[i][l] = 0
                    elif self.policy == 'ones':
                        labels[i][l]=1  # all ones
                    elif self.policy == 'zeros': # all zeros
                        labels[i][l] == 0
                else:
                    labels[i][l] = 0
        self.labels = labels
        self.images = image_names
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        '''Take the index of item and returns the image and its labels'''
        img = self.images[index]
        img = Image.open(img).convert('RGB')
        label = self.labels[index]
        
        if self.split=='train':
            transform = transforms.Compose([
                transforms.Resize(self.target_shape),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.target_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            
        img = transform(img)
        
        return img, label, index
    
class CheXpert(MyDatasetBase):
    def __init__(self, dataset_dir, idx, policy, target_shape):
        super().__init__(dataset_dir)
        self.target_shape = target_shape
        self.idx = idx
        self.policy = policy
        
    def get_datasets(self):
        path = os.path.abspath(self.dataset_dir)
        train_dataset = CheXpertDataset(path, self.idx, self.policy, target_shape=
                                       self.target_shape, split='train')
        valid_dataset = CheXpertDataset(path, self.idx, self.policy, target_shape=
                                        self.target_shape, split='valid')
        test_dataset = CheXpertDataset(path, self.idx, self.policy, target_shape= 
                                      self.target_shape, split = 'test')
        return train_dataset, valid_dataset, test_dataset
    
    def get_data_loaders(self, batch_size):
        train_dataset, valid_dataset, test_dataset = self.get_datasets()
        train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size,
                                 shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size = batch_size, 
                                  shuffle=False)
        test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size,
                                shuffle=True)
        return train_loader, valid_loader, test_loader







