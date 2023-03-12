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
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)



        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)


        x = self.flatten(x)
        # print("flatten")
        # print(x.shape)
        x = self.dense1(x)
        # print("dense1")
        # print(x.shape)
        x = self.relu4(x)

        x = self.dense2(x)

        return x


class AACN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 10 * 10, 512)
        self.actor = nn.Linear(512, 4)
        self.softmax = nn.Softmax(dim=-1)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        actor_output = self.actor(x)
        actor_output = self.softmax(actor_output)
        critic_output = self.critic(x)

        return actor_output, critic_output