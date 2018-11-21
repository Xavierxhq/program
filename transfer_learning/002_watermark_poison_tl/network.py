import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms


class Network(nn.Module):
    def __init__(self, layers=4):
        super(Network, self).__init__()
        self.layers = layers

        if self.layers == 4:
            self.conv1 = nn.Sequential( # (1,28,28)
                         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                                   stride=1, padding=2), # (16,28,28)
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2)
                         )
            self.conv2 = nn.Sequential( # (16,14,14) -> (16,28,28)
                         nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14) -> (32,28,28)
                         nn.ReLU(),
                         nn.MaxPool2d(2)
                         )
            self.fc1 = nn.Linear(32*7*7, 1000)
        elif self.layers == 6:
            self.conv1 = nn.Sequential( # (1,28,28)
                         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                                   stride=1, padding=2), # (16,28,28)
                         nn.ReLU(),
                         )
            self.conv2 = nn.Sequential( # (16,14,14) -> (16,28,28)
                         nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14) -> (32,28,28)
                         nn.ReLU(),
                         )
            self.conv3 = nn.Sequential( # (32,28,28)
                         nn.Conv2d(32, 64, 5, 1, 2), # (64,28,28)
                         nn.ReLU(),
                         nn.MaxPool2d(2) # (64,14,14)
                         )
            self.conv4 = nn.Sequential( # (64,14,14)
                         nn.Conv2d(64, 64, 5, 1, 2), # (64,14,14)
                         nn.ReLU(),
                         nn.MaxPool2d(2) # (64,7,7)
                         )
            self.fc1 = nn.Linear(64*7*7, 1000)
        self.out = nn.Linear(1000, 10)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x): # this functions is not for external call
        x = self.conv1(x)
        x = self.conv2(x)
        if self.layers == 6:
            x = self.conv3(x)
            x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.out(x)
        return output

    def init_optimizer(self, model_params=None, learning_rate=None):
        if model_params is None or learning_rate is None:
            print('model_params or learning_rate not provided.')
            exit(100)
        self.optimizer = torch.optim.Adam(model_params, lr=learning_rate)

    def init_transfer_optimizer(self, model_params=None, learning_rate=None):
        if model_params is None or learning_rate is None:
            print('model_params or learning_rate not provided.')
            exit(100)
        params = filter(lambda p: p.requires_grad, model_params)
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def calc_loss(self, output=None, y=None):
        if output is None or y is None:
            print('output or y not provided.')
            exit(100)
        loss = self.loss_func(output, y)
        return loss

    def backward(self, loss=None):
        if self.optimizer is None:
            print('optimizer not initiated.')
            exit(200)
        if loss is None:
            print('loss not provided.')
            exit(100)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
