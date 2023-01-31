#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# In[4]:

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p
        self.seed = 0
        
    def forward(self, input, freeze = False):
        if not self.training:
            return input
        
        if not freeze:
            q = np.random.randint(10000000, size=1)[0]
            self.seed = q
            
        torch.manual_seed(self.seed)
        return torch.nn.functional.dropout2d(input, p = self.p)


class ConvNet(nn.Module):
    def __init__(self, in_channels, num_classes, droprate=0.5):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 1, padding = 1), # input size = 32, conv2d = 32, conv2d = 32, maxpool2d = 16
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        #nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        
        self.layer2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding = 1),   # input size = 16, conv2d = 16, conv2d = 16, maxpooling = 8
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, stride =1, padding = 1),
        nn.ReLU(),
        #nn.BatchNorm2d
        MyDropout(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.layer3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), # input_size = 8, conv2d = 8, conv2d = 8, maxpool = 4
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, stride =1, padding = 1),
        nn.ReLU(),
        MyDropout(),
        nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256*4*4, 240),
        nn.ReLU(),
        MyDropout(),
        nn.Linear(240, num_classes)
        )
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = x.view(x.size(0), -1)
        x = self.fc(x)
        return(x)
        






