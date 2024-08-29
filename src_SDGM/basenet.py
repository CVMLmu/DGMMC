import torch
import torch.nn as nn
import torch.nn.functional as F
from .SDGM import SDGM


class simpleCNN_SDGM(nn.Module):
    def __init__(self, features_dim, n_class, n_component=1, cov_type="full"):
        super(simpleCNN_SDGM, self).__init__()
        self.n_class = n_class
        self.n_component = n_component
        self.cov_type = cov_type
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.fc_rd = nn.Linear(1152, features_dim)
        self.last = SDGM(features_dim, self.n_class, self.n_component, self.cov_type)

    def forward(self, x, target=None):
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
        x = self.last(x)
        return x

class simpleCNN_SDGM_relu(nn.Module):
    def __init__(self, features_dim, n_class, n_component=1, cov_type="full"):
        super(simpleCNN_SDGM_relu, self).__init__()
        self.n_class = n_class
        self.n_component = n_component
        self.cov_type = cov_type
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.fc_rd = nn.Linear(1152, features_dim)
        self.last = SDGM(features_dim, self.n_class, self.n_component, self.cov_type)

    def forward(self, x, target=None):
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
        #x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_rd(x)
        x = F.relu(x)
        x = self.last(x)
        return x