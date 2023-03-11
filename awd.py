import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.relu3 = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(in_features=1024, out_features=64)
        self.relu4 = nn.ReLU(inplace=True)

        self.dense2 = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        # print("conv1")
        # print(x.shape)

        x = self.conv2(x)
        x = self.relu2(x)

        # print("conv2")
        # print(x.shape)

        x = self.conv3(x)
        x = self.relu3(x)

        # print("conv3")
        # print(x.shape)

        x = self.flatten(x)
        # print("flatten")
        # print(x.shape)
        x = self.dense1(x)
        # print("dense1")
        # print(x.shape)
        x = self.relu4(x)

        x = self.dense2(x)

        return x

