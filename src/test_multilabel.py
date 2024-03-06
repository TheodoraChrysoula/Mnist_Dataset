#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Models
import ResNet50_Model 

from tqdm import tqdm 

import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader


# In[39]:


def test(model, testloader, device):
    #testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    #model = model.model(pretrained=True, requires_grad=False, num_classes)
    #checkpoint=torch.load(model_file)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('Testing the model')
    
    counter=0.
    #predictions=[]
    true_labels=[]
    idxs=[]
    label_indices=[]
    preds=[]
    
    with torch.no_grad():
        
        for image, label, bbox, idx in tqdm(testloader):
            counter+=1
            image=image.to(device)
            label=label.to(device)
            #label = label.to(device)
            idx = idx.detach().cpu().numpy().astype('int')
            idxs.append(idx)
            # forward pass
            output=model(image)
            pred = torch.sigmoid(output)[0].detach().cpu().numpy().astype('float')
            preds.append(pred)
            label = label.detach().cpu().numpy().astype('int')
            true_labels.append(label)
            indices = [i for i in range(len(label[0])) if label[0][i]==1]
            label_indices.append(indices)

        return preds, true_labels, idxs, label_indices

    


# In[ ]:




