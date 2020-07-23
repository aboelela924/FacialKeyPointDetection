import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Network(nn.Module):
    def __init__(self, ):
        super(Network,self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, 3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(128, 256, 3)
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.conv4 = torch.nn.Conv2d(256, 512, 3)
        self.pool4 = torch.nn.MaxPool2d(2, 2)
        self.conv5 = torch.nn.Conv2d(512, 1024, 3)
        self.pool5 = torch.nn.MaxPool2d(2, 2)
        self.conv6 = torch.nn.Conv2d(1024, 2048, 3)
        self.pool6 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(2048, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc3 = torch.nn.Linear(256, 136)
        self.drop1 = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))

        x = self.drop1(x)

        x = x.view(x.size(0), -1)

        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop1(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x