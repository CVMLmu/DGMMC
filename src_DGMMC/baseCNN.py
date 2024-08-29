import torch
import torch.nn as nn
import torch.nn.functional as F


class simpleCNN(nn.Module):
    def __init__(self, output_dimension = 2):
        super(simpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.fc_rd = nn.Linear(1152, output_dimension)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_rd(x)
        return x