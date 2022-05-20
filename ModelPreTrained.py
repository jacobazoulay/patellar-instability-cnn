# pytorch
# Baselines: 1 hidden layer NN | 5 layer CNN
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class KeypointPretrainedModel(nn.Module):
    def __init__(self, args):
        super(KeypointPretrainedModel, self).__init__()

        #self.model = torchvision.models.vgg16() # loads model architecture with random weights
        self.model = torchvision.models.vgg16(pretrained = True) # loads model architecture with pretrained weights
        #self.model = torchvision.models.vgg16_bn(pretrained = True) # loads model architecture (with batch norm) with pretrained weights

        for param in self.model.parameters(): # Freeze model weights (remove to unfreeze all weights)
            param.requires_grad = False
        
        numFeatures = self.model.fc.in_features
        self.fc = nn.Linear(numFeatures, 6) # output is # of coordinates (3 key points)

        #self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # takes in 1 input image channel
        #self.conv2 = nn.Conv2d(6, 16, kernel_size=3)

        #self.fc1 = nn.Linear(16 * 30 * 30, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 6) #output is # of coordinates (3 keypoints)

        #self.pool = nn.MaxPool2d(2, 2)

        
    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        out = self.fc(x)
        
        return out
