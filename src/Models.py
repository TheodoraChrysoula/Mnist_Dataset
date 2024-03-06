#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as torch_models

def model(model_type, pretrained, requires_grad, res, num_classes):
    if model_type=='resnet':
        
        if pretrained:
            if res == 18:
                resnet = torch_models.resnet18(weights = torch_models.ResNet18_Weights.DEFAULT)
            elif res == 50:
                resnet = torch_models.resnet50(weights = torch_models.ResNet50_Weights.DEFAULT)
            elif res == 101:
                resnet = torch_models.resnet101(weights = torch_models.ResNet101_Weights.DEFAULT)
        else:
            resnet = torch_models.resnet18(weights = None)
            
    
        # Replace inplace ReLU with regular ReLU
        for module in resnet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace=False


        # to freeze the hidden layers
        if requires_grad == False:
            for param in resnet.parameters():
                param.requires_grad=False

        # to train the hidden layers
        elif requires_grad == True:
            for param in resnet.parameters():
                param.requires_grad=True
            
        # make the classification layer learnable, i have 7 classes

        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes))
        return resnet
    
    elif model_type == 'densenet':
        if pretrained:
            if res==121:
                densenet = torch_models.densenet121(pretrained=True)
            elif res == 169:
                densenet = torch_models.densenet169(pretrained=True)
            elif res == 201:
                densenet = torch_models.densenet201(pretrained=True)
        else:
            densenet = torch.models.densenet121(pretrained=False)
            
        # Replace inplace ReLU with regular ReLU
        for module in densenet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
                
        # Freeze or train the hidden layers
        if not requires_grad:
            for param in densenet.parameters():
                param.requires_grad=False
        else:
            for param in densenet.parameters():
                param.requires_grad=True
                
        # Modify the classification layer for the desired number of classes
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, num_classes)
        
        return densenet
    
    elif model_type == 'efficient':
        efficient = torch_models.efficientnet_b0(pretrained=True)
        
        # Freeze or train the hiden parameters
        for param in efficient.parameters():
            param.requires_grad=True
            
        # Modify the classification layer for the desired number of classes
        num_ftrs = efficient.classifier[1].in_features
        efficient.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        return efficient
                



