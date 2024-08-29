import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_ard import LinearARD


class SDGM(nn.Module):
    def __init__(self, input_dim, n_class, n_component=1, cov_type="diag", **kwargs):
        """An SDGM layer, which can be used as a last layer for a classificaiton neural network.
        Attributes:
            input_dim (int): Input dimension
            n_class (int): The number of classes
            n_component (int): The number of Gaussian components
            cov_type: (str): The type of covariance matrices. If "diag", diagonal matrices are used, which is computationally advantageous. If "full", the model uses full rank matrices that have high expression capability at the cost of increased computational complexity.
        """
        super(SDGM, self).__init__(**kwargs)
        assert input_dim > 0
        assert n_class > 1
        assert n_component > 0
        assert cov_type in ["diag", "full"]
        self.input_dim = input_dim
        self.n_class = n_class
        self.n_component = n_component
        self.cov_type = cov_type
        self.n_total_component = n_component*n_class
        self.ones_mask = (torch.triu(torch.ones(input_dim, input_dim)) == 1)
        # Bias term will be set in the linear layer so we omitted "+1"
        if cov_type == "diag":
            self.H = int(2 * self.input_dim)
        else:
            self.H = int(self.input_dim * (self.input_dim + 3) / 2)
        # Network
        self.fc = LinearARD(self.H, self.n_total_component)

    def forward(self, x):
        x = self.nonlinear_transformation(x)
        x = self.fc(x)
        x = torch.reshape(x, (-1, self.n_class, self.n_component))
        output = torch.logsumexp(x, dim=-1)
        return output

    def nonlinear_transformation(self, x):
        if self.input_dim == 1 or self.cov_type == "diag":
            quadratic_term = x*x
        else:
            outer_prod = torch.einsum('ni,nj->nij', x, x)
            quadratic_term = outer_prod[:, self.ones_mask]
        # bias_term will be set in the next linear layer
        output = torch.cat([x, quadratic_term], dim=1)
        return output
