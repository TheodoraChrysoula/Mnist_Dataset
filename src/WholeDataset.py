#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import os
from sklearn.model_selection import train_test_split
from skimage import io, transform
import PIL


# In[2]:


DATA_DIR = '../vinbigdata-chestXRay/'
#DATA_DIR


# In[3]:


#root_dir = DATA_DIR+'dataset_no.csv'
boxes_dir = DATA_DIR+'converted_data.csv'
images_dir = DATA_DIR+'converted'


# In[4]:


class ChestDataset(Dataset):
    def __init__(self, root_dir, images_dir, boxes_dir, train, test):
        '''
        Args:
             csv_file (string): Path to csv file with annotations.
             root_dir (string): Directory with all the images.
             boxes (string): Path to csv file that contains the boxes
             train: true -> prepare the train dataset
             test: True -> prepare the test dataset
        '''
        
        self.csv = pd.read_csv(root_dir)
        #print(self.csv.shape)
        self.boxes = pd.read_csv(boxes_dir)
        #print(self.boxes.shape)
        self.images_dir = images_dir
        #print(self.images_dir)
        self.train = train
        self.test = test
        self.images = self.csv[:]['image_path'].values  # take the image path of the pictures (numpy.ndarray)
        #print(self.images[0])
        self.labels = np.array(self.csv.drop(['image_id', 'image_path'], axis=1)) # take the label of each image. shape: (15000, 14)
        #print(self.labels[0])
        self.boxes = self.boxes[['image_path', 'x_min', 'y_min', 'x_max', 'y_max']]
        #print(self.boxes.shape)
        
        self.train_ratio = int(0.8*len(self.csv))
        #print(self.train_ratio) # train ratio: 12000 images
        self.valid_test_ratio = len(self.csv)-self.train_ratio # valid and test ratio: 3000 images
        #print(self.valid_test_ratio)
        self.valid_ratio = int(0.5*self.valid_test_ratio) # valid ratio: 1500
        #print(self.valid_ratio)
        self.test_ratio = self.valid_test_ratio-self.valid_ratio # test ratio: 1500
        #print(self.test_ratio)
        
        # set the training data images and labels
        if self.train==True and self.test==False:
            
            print(f"The number of training images: {self.train_ratio}")
            
            self.image_names = list(self.images[:self.train_ratio])
            #print(len(self.image_names))
            #print(self.image_names[0])
            self.labels = self.labels[:self.train_ratio]
            #print(len(self.labels))
            #print(self.labels[0])
            #self.boxes = self.boxes[self.train_ratio:][:]
            
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.485, 0.456, 0.406]),    # imagenet mean and std
                                     std=([0.229, 0.224, 0.225]))
            ])
            
        # set the validation data images and labels
        
        elif self.train==False and self.test==False:
            print(f"The number of validation images: {self.valid_ratio}")
            
            self.image_names = list(self.images[self.train_ratio:(self.train_ratio + self.valid_ratio)])
            self.labels = self.labels[self.train_ratio:(self.train_ratio+self.valid_ratio)]
                                                
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.485, 0.456, 0.406]),
                                    std=([0.229, 0.224, 0.225]))
            ])
            
        # set the test data images and labels
        elif self.test == True and self.train==False:
            print(f"The number of the test images: {self.test_ratio}")
            self.image_names = list(self.images[-self.test_ratio:])
            self.labels = self.labels[-self.test_ratio:]
            # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.485, 0.456, 0.406]),
                                    std = ([0.229, 0.224, 0.225]))
                
            ])
        
        
    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        if self.train==True or self.test==False:
            img_name = self.image_names[idx]
            image = cv2.imread(img_name)
            # convert the image from BGR to RGB for matplotlib
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # apply the image transforms
            image = self.transform(image)
            targets = self.labels[idx]
            return image, targets, idx
        
        elif self.train == False and self.test==True:
            img_name = self.image_names[idx]
            image = cv2.imread(img_name)
            # Convert the image from BGR to RGB for matplotlib
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # apply the image transforms
            image = self.transform(image)
            targets = self.labels[idx]
            bbox = self.boxes.loc[self.boxes['image_path']==img_name, ['x_min', 'y_min', 'x_max', 'y_max']].values
            return image, targets,  bbox, idx
        
        
#         img_name = self.image_names[idx]
#         print(img_name)
#         image = cv2.imread(img_name)
#         # convert the image from BGR to RGB for matplotlib
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # apply the image transforms
#         image = self.transform(image)
#         targets = self.labels[idx]
#         #box = self.boxes.loc[self.boxes['image_path']==img_name, ['x_min', 'y_min', 'x_max', 'y_max']].values
            
#         return image, targets, idx 
    
def dataset_whole(images_dir, boxes_dir, num_classes):
    if num_classes == 13:
        root_dir = DATA_DIR + 'data_13c.csv'
    else:
        root_dir = DATA_DIR + 'dataset_no.csv'
    train_dataset=ChestDataset(root_dir, images_dir, boxes_dir, train=True, test=False)
    valid_dataset=ChestDataset(root_dir, images_dir, boxes_dir, train=False, test=False)
    test_dataset= ChestDataset(root_dir, images_dir, boxes_dir, train=False, test=True)
    return train_dataset, valid_dataset, test_dataset




