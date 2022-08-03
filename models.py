from turtle import forward
from PIL import Image
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
import albumentations.pytorch
from matplotlib import pyplot as plt
import cv2
import numpy as np
import argparse
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
import os

class custom_model(nn.Module):
    """define custom model for classifcation"""
    def __init__(self, num_classes, img_shape):
        super().__init__()
        self.kernel_size = 3
        self.pool_size = 2
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.flatten = None
        self.conv1 = nn.Conv2d(3, 32, self.kernel_size)
        self.b1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, self.kernel_size)
        self.b2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, self.kernel_size)
        self.b3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, self.kernel_size)
        self.b4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, self.kernel_size)
        self.b5 = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(self.pool_size, self.pool_size)
        print(self.flatten)
        # self.r = self.flatten(*self.flatten(*self.flatten(w=self.img_shape, k=self.kernel_size, s=1, p=0, m=True)))[0]
        self.fc1 = nn.Linear(128*5*5, 256)
        self.d1 = nn.Dropout(0.5)
        self.out = nn.Linear(256, self.num_classes)
    
    def forward(self, x):
        x = self.pool(self.b1(F.relu(self.conv1(x))))
        x = self.pool(self.b2(F.relu(self.conv2(x))))
        x = self.pool(self.b3(F.relu(self.conv3(x))))
        x = self.pool(self.b4(F.relu(self.conv4(x))))
        x = self.pool(self.b5(F.relu(self.conv5(x))))
        x = torch.flatten(x, 1)
        self.flatten = x.shape
        x = self.d1(F.relu(self.fc1(x)))
        output = self.out(x)
        return output

    #for automatic fc neurons calculations after convolutions and pooling operations
    def flatten(self, w, k=3, s=1, p=0, m=True):
        """
        Returns the right size of the flattened tensor after
            convolutional transformation
        :param w: width of image
        :param k: kernel size
        :param s: stride
        :param p: padding
        :param m: max pooling (bool)
        :return: proper shape and params: use x * x * previous_out_channels

        Example:
        r = flatten(*flatten(*flatten(w=100, k=3, s=1, p=0, m=True)))[0]
        self.fc1 = nn.Linear(r*r*128, 1024)
        """
        return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m

model= custom_model(2, 250)
print(model)
