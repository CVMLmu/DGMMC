# SDGM in Python
Code for jointly training a deep neural network with the SDGM as the last layer

## Usage
Import the SDGM class when you build a network.
```
from SDGM import SDGM
```

The SDGM class can be used in a similar way as torch.nn.Linear.
```
# In _init__() function
self.n_class = ... # Number of classes
self.n_component = ... # Number of Gaussian components
self.cov_type = ... # Covariance type ("diag" or "full)
self.last = SDGM(2, self.n_class, self.n_component, self.cov_type)
# In forward() function
x = self.last(x)
```

The ELBOLoss class in torch_arg.py is required for training. 
```
from torch_ard import ELBOLoss
```

In the main loop of training, ELBOLoss is used instead of the corss entropy. 
```
def get_kl_weight(epoch, max_epoch): return min(1, 1e-9 * epoch / max_epoch)
criterion = ELBOLoss(model, F.cross_entropy).to("cuda")
for epoch in ...:
    kl_weight = get_kl_weight(epoch, args.n_epochs)
    loss = criterion(outputs, labels, 1, kl_weight)
```

For practical usage, please refer to the demo directory that includes MNIST classification. 