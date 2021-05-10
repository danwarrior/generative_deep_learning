import numpy as np

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv_1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)  # [N, 3, 32, 32] => [N, 32, 32, 32]
        self.bn_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.LeakyReLU(0.1)

        self.conv_2 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)  # [N, 32, 32, 32] => [N, 32, 16, 16]
        self.bn_2 = nn.BatchNorm2d(32)
        self.relu_2 = nn.LeakyReLU(0.1)

        self.conv_3 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)  # [N, 32, 16, 16] => [N, 64, 16, 16]
        self.bn_3 = nn.BatchNorm2d(64)
        self.relu_3 = nn.LeakyReLU(0.1)

        self.conv_4 = nn.Conv2d(64, 64, (3, 3), stride=2, padding=1)  # [N, 64, 16, 16] => [N, 64, 8, 8]
        self.bn_4 = nn.BatchNorm2d(64)
        self.relu_4 = nn.LeakyReLU(0.1)

        self.flatten = nn.Flatten()  # [N, 64, 8, 8] => [N, 64*8*8]

        self.forward_block = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # [N, 64, 8, 8] => [N, 128]
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 10),  # [N, 128] => [N, 10]
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)

        x = self.flatten(x)
        x = self.forward_block(x)

        return x