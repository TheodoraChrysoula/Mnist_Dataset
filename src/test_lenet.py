#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tqdm import tqdm 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import medmnist 
from medmnist import INFO, Evaluator

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score


# In[5]:


from dataset import train_dataset, test_dataset, n_classes, n_channels, class_names, test_loader, task


# In[6]:


from Model_LeNet import LeNet



# In[7]:


print(task)


# In[8]:


model = LeNet(n_channels, n_classes)
model.load_state_dict(torch.load('../outputs/mnist_pneumonia_lenet.ckpt', map_location=torch.device('cpu')))


# In[9]:


def save_test_results(tensor, labels, output_class, counter):
    '''
    This function will save a few test images along with the
    ground truth label and predicted label annotated on the image
    
    :param tensor: the image tensor
    :param target: the ground truth class
    param output_class: the predicted class number
    param counter: the test image number
    '''
    
    # Move tensor to cpu and denormalize
    image = torch.squeeze(tensor, 0).cpu().numpy()
    image = image/2 + 0.5
    image = np.transpose(image, (1,2,0))
    # Convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt = labels.cpu().numpy()
    cv2.putText(
    image, f"GT: {gt}",
    (5,25), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
    image, f"Pred: {output_class}",
    (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 
    0.7, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.imwrite(f"../outputs/test_image_{counter}.png", image*255.)
    


# In[10]:


def test(model, testloader, task):
    model.eval()
    print('Testing the model')
    prediction_list = []
    ground_truth_list = []
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total = len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(image)
            if task == 'multi-label, binary-class':
                predictions = torch.sigmoid(outputs, 1).cpu().numpy()
            else:
                predictions = F.softmax(outputs, 1).cpu().numpy()
            output_class = np.argmax(predictions)

            # Append the GT and predictions to the respective lists
            prediction_list.append(output_class)
            ground_truth_list.append(labels.cpu().detach().numpy().flatten())
            # Calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels.squeeze().long()).sum().item()
            # Save few test images
            if counter % 99 == 0:
                save_test_results(image, labels, output_class, counter)
        acc = 100.*(test_running_correct/ len(testloader.dataset))
        return prediction_list, ground_truth_list, acc
                


# In[11]:


if __name__ == '__main__':
    
    model = LeNet(n_channels, n_classes)  # LeNet accyracy: 89.075
    print(model.__class__.__name__ )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load('../outputs/mnist_pneumonia_lenet.ckpt', map_location=torch.device('cpu')))
    
    prediction_list, ground_truth_list, acc = test(model, test_loader, task)
    print(f"Test accuracy: {acc:.3f}%")
    
   
    
    # Confusion matrix
    conf_matrix = confusion_matrix(ground_truth_list, prediction_list)
    
    plt.figure(figsize=(12,9))

    sns.heatmap(
    conf_matrix, 
    annot = True,
    xticklabels = class_names,
    yticklabels = class_names)
    
    plt.savefig('../outputs/heatmaps_LeNet_Pneumonia.png')
    plt.close()


# In[12]:


model = LeNet(n_channels, n_classes)  # LeNet accyracy: 89.075
print(model.__class__.__name__ )
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load('../outputs/mnist_pneumonia_lenet.ckpt', map_location=torch.device('cpu')))

prediction_list, ground_truth_list, acc = test(model, test_loader, task)
print(f"Test accuracy: {acc:.3f}%")


# In[ ]:




