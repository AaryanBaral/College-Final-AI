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
    def make_blocks(self, block:ResBlock, out_channels:int, num_blocks:int, stride:int):
        '''make a residual block'''
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in strides:
            layers.append(block(self.in_channels, out_channels, stride=i))
            self.in_channels=out_channels
        return nn.Sequential(*layers)

    def forward(self, x:torch.tensor)->torch.tensor:
        out = self.batchnorm1(self.conv1(x))
        out = F.max_pool2d(torch.relu(out), 2)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.layer1(out)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.layer2(out)
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.layer3(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        return out
# ## Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), output, target_class