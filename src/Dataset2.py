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


root_dir = DATA_DIR+'dataset_no.csv'
boxes_dir = DATA_DIR+'converted_data.csv'
images_dir = DATA_DIR+'converted'


# In[4]:


data = pd.read_csv(root_dir)


# In[5]:


def data_csv(data, num_classes, title, no_finding=True,):
    data1 = copy.deepcopy(data)
    if num_classes==3:
        class_1 = data1[data1["0"]==1].head(1200)
        class_2 = data1[data1["3"]==1].head(1200)
        class_3 = data1[data1["14"]==1].head(1200)
        data1 = pd.concat([class_1, class_2, class_3])
        i_shape = data1.shape[0]
        print(i_shape)
        data1 = data1.drop_duplicates()
        data1 = data1.sample(frac=1)
        data1 = data1[["image_id", "image_path", "0", "3", "14"]]
        f_shape = data1.shape[0]
        print(f_shape)
        print(f"The number of the common images is: {i_shape-f_shape}")
        data1.to_csv(title, index=False)
        return data1
    elif num_classes == 4:
        if no_finding==True:
            class_1 = data1[data1["0"]==1].head(1200)
            class_2 = data1[data1["3"]==1].head(1200)
            class_3 = data1[data1["11"]==1].head(1200)
            class_4 = data1[data1["14"]==1].head(1200)
            data1 = pd.concat([class_1, class_2, class_3, class_4])
            i_shape = data1.shape[0]
            print(i_shape)
            data1 = data1.drop_duplicates()
            data1 = data1.sample(frac=1)
            data1 = data1[["image_id", "image_path", "0", "3", "11", "14"]]
            f_shape = data1.shape[0]
            print(f_shape)
            print(f"The number of common images is: {i_shape-f_shape}")
            data1.to_csv(title, index=False)
            return data1
        
        else:
            class_1 = data1[data1["0"]==1].head(1200)
            class_2 = data1[data1["3"]==1].head(1200)
            class_3 = data1[data1["11"]==1].head(1200)
            class_4 = data1[data1["13"]==1].head(1200)
            data1 = pd.concat([class_1, class_2, class_3, class_4])
            i_shape = data1.shape[0]
            print(i_shape)
            data1 = data1.drop_duplicates()
            data1 = data1.sample(frac=1)
            data1 = data1[["image_id", "image_path", "0", "3", "11", "13"]]
            f_shape = data1.shape[0]
            [print(f_shape)]
            print(f"The number of common images is: {i_shape-f_shape}")
            data1.to_csv(title, index=False)
            return data1
        
    elif num_classes == 5:
        class_1 = data1[data1["0"]==1].head(1200)
        class_2 = data1[data1["3"]==1].head(1200)
        class_3 = data1[data1["11"]==1].head(1200)
        class_4 = data1[data1["13"]==1].head(1200)
        class_5 = data1[data1["14"]==1].head(1200)
        data1 = pd.concat([class_1, class_2, class_3, class_4, class_5])
        i_shape = data1.shape[0]
        print(i_shape)
        data1 = data1.drop_duplicates()
        data1 = data1.sample(frac=1)
        data1 = data1[["image_id", "image_path", "0", "3", "11", "13", "14"]]
        f_shape = data1.shape[0]
        print(f_shape)
        print(f"The number of common images is: {i_shape-f_shape}")
        data1.to_csv(title, index=False)
        return data1
    elif num_classes == 7:
        class_1 = data1[data1["0"]==1]
        class_2 = data1[data1["1"]==1]
        class_3 = data1[data1["7"]==1]
        class_4 = data1[data1["10"]==1]
        class_5 = data1[data1["11"]==1]
        class_6 = data1[data1["13"]==1]
        class_7 = data1[data1["14"]==1]
        data1 = pd.concat([class_1, class_2, class_3, class_4, class_5, class_6, class_7])
        i_shape = data1.shape[0]
        data1 = data1.drop_duplicates()
        data1 = data1.sample(frac=1)
        data1 = data1[['image_id', 'image_path', '0', '3', '7', '10', '11', '13', '14']]
        f_shape = data1.shape[0]
        print(f"The number of common images is: {i_shape-f_shape}")
        data1.to_csv(title, index=False)
        return data1
    elif num_classes == 10:
        class_1 = data1[data1["0"]==1]
        class_2 = data1[data1["3"]==1]
        class_3 = data1[data1["6"]==1]
        class_4 = data1[data1["7"]==1]
        class_5 = data1[data1["8"]==1]
        class_6 = data1[data1["9"]==1]
        class_7 = data1[data1["10"]==1]
        class_8 = data1[data1["11"]==1]
        class_9 = data1[data1["13"]==1]
        class_10 = data1[data1["14"]==1]
        data1 = pd.concat([class_1, class_2, class_3, class_4, class_5, class_6, class_7, 
                           class_8, class_9, class_10])
        i_shape = data1.shape[0]
        data1 = data1.drop_duplicates()
        data1 = data1.sample(frac=1)
        data1 = data1['image_id', 'image_path', '0', '3', '6', '7', '8', '9', '10', '11', 
                      '13', '14']
        f_shape = data1.shape[0]
        print(f"The number of common images is: {i_shape-f_shape}")
        data1.to_csv(title, index=False)
        return data1
              
    


# In[7]:


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
    


# In[12]:


def dataset(num_classes, images_dir, boxes_dir, no_finding=True):
    if num_classes==3:
        root_dir = DATA_DIR + "data_3c.csv"
    elif num_classes==4 and no_finding==True:
        root_dir = DATA_DIR + "data_4c.csv"
    elif num_classes==4 and no_finding==False:
        root_dir = DATA_DIR + "data_4cno.csv"
    elif num_classes == 5:
        root_dir = DATA_DIR + "data_5c.csv"
    elif num_classes == 7:
        root_dir = DATA_DIR + "data_7c.csv"
    elif num_classes == 10:
        root_dir = DATA_DIR + "data_10c.csv"
    
    train_dataset = ChestDataset(root_dir, images_dir, boxes_dir, train=True, test=False)
    valid_dataset = ChestDataset(root_dir, images_dir, boxes_dir, train=False, test=False)
    test_dataset = ChestDataset(root_dir, images_dir, boxes_dir, train=False, test=True)
    
    return train_dataset, valid_dataset, test_dataset
                    
        


# In[17]:


#train_dataset, valid_dataset, test_dataset = dataset(4, images_dir, boxes_dir, no_finding=True)


# In[23]:


# data = pd.read_csv(DATA_DIR+"dataset_no.csv")
# train_data = data[:12000]
# valid_data = data[12000:13500]
# test_data = data[-1500:]
# print(train_data.shape, valid_data.shape, test_data.shape)


# # In[24]:


# labels = {0: 'Aortic enlargement',
#          1: 'Atelectasis',
#          2: 'Calcification',
#          3: 'Cardiomegaly',
#          4: 'Consolidation',
#          5: 'ILD', 
#          6: 'Infiltration',
#          7: 'Lung opacity',
#          8: 'Nodule/Mass',
#          9: 'Other lesion',
#          10: 'Pleural effusion',
#          11: 'Pleural thickening',
#          12: 'Pneumothorax',
#          13: 'Pulmonary fibrosis',
#          14: 'No_finding'}


# # In[25]:


# def plot_labels(dataframe, title=''):
#     sns.set(font_scale = 2)
#     plt.figure(figsize=(12, 8))

#     #sns.load_dataset(multilabel_counts.all())
#     x = dataframe.index.values
#     #print(x.shape)
#     y = dataframe[0].values
#     #print(y)
#     ax = sns.barplot(data=dataframe, x=x, y=y)

#     plt.title(title)
#     plt.ylabel('Number of images', fontsize=18)
#     plt.xlabel('Number of labels', fontsize=18)

#     #adding the text labels
#     rects = ax.patches
#     labels = dataframe.values
#     #print(labels)
#     for rect, label in zip(rects, labels):
#         height = rect.get_height()
#         ax.text(rect.get_x()+rect.get_width()/2, height+5, label, ha='center', va='bottom')

#     plt.show()


# # In[26]:


# summed_train_data = train_data.iloc[:,2:17].sum()
# summed_train_data = pd.DataFrame(summed_train_data).reset_index(drop=True)
# plot_labels(summed_train_data, 'Train images having multiple label')


# # In[27]:


# summed_valid_data = valid_data.iloc[:,2:17].sum()
# summed_valid_data = pd.DataFrame(summed_valid_data).reset_index(drop=True)
# plot_labels(summed_valid_data, 'Valid images having multiple label')


# # In[28]:


# summed_test_data = test_data.iloc[:,2:17].sum()
# summed_test_data = pd.DataFrame(summed_test_data).reset_index(drop=True)
# plot_labels(summed_test_data, 'Test images having multiple label')


# # In[ ]:




