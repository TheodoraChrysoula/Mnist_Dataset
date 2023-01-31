#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



# In[15]:


class MyDropout(nn.Module):
    def __init__(self, p = 0.5):
        super(MyDropout, self).__init__()
        self.p = p
        self.seed = 0
        
    def forward(self, input, freeze=False):
        if not self.training:
            return input
        
        if not freeze:
            q = np.random.randint(1000000, size=1)[0]
            self.seed = q
        
        torch.manual_seed(self.seed)
        return torch.nn.functional.dropout2d(input, p = self.p)


# In[16]:


class LeNet(nn.Module):
    def __init__(self, input_channels, num_classes, droprate=0.5):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(input_channels,24,5),
        nn.ReLU(),
        nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
        nn.Conv2d(24, 48, 5),
        nn.ReLU(),
        MyDropout(),
        nn.MaxPool2d(2))

        self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(48*5*5, 240),
        nn.ReLU(),
        MyDropout(),
        nn.Linear(240, num_classes))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x
    






