import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from glob import glob
# ## Model and Device Setup

model_path = "model_final_weights.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ## Disease Classes

diseases = ['cataract','diabetic_retinopathy','glaucoma', 'normal']
class ResBlock(nn.Module):
    '''A resnet block with skip connection'''
    def __init__(self, in_channels:int, out_channels:int, stride:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if (in_channels!=out_channels or stride!=1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x:torch.tensor)->torch.tensor:
        out = torch.relu(self.batchnorm1(self.conv1(x)))
        out = self.batchnorm2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

        
class ResNet18(nn.Module):
    '''A ResNet18 model'''
    def __init__(self, num_classes:int=8):  # Set to your number of classes
        super().__init__()
        self.in_channels=64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)

        self.layer1 = self.make_blocks(ResBlock, 64, 2, 1)
        self.layer2 = self.make_blocks(ResBlock, 128, 2, 2)
        self.layer3 = self.make_blocks(ResBlock, 256, 2, 2)
        self.layer4 = self.make_blocks(ResBlock, 512, 2, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)