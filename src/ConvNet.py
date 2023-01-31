#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# In[3]:


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p
        self.seed = 0
        
    def forward(self, input, freeze = False):
        if not self.training:
            return input
        
        if not freeze:
            q = np.random.randint(10000000, size = 1)[0]
            self.seed = q
            
        torch.manual_seed(self.seed)
        return torch.nn.functional.dropout2d(input, p = self.p)
    


# In[4]:


class ConvNet(nn.Module):
    def __init__(self, in_channels, num_classes, droprate = 0.5):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, 1, 1), #input_size = 32, conv2d = 32, maxpool = 16
        nn.ReLU(),
        MyDropout(),
        nn.MaxPool2d(2,2)
        )
        
        self.layer2 = nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, 1), # input_size = 16, conv2d = 16, maxpool = 8
        nn.ReLU(),
        MyDropout(),
        nn.MaxPool2d(2,2)
        )
        
        self.layer3 = nn.Sequential(
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        MyDropout(),
        nn.MaxPool2d(2,2)
        )
        
        self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128*4*4, 128),
        nn.ReLU(),
        MyDropout(),
        nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return(x)
        
        


# In[ ]:




