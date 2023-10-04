import numpy as np
import torch

x = torch.tensor([1, 2, 3, 4])
print(x)
torch.unsqueeze(x, 1)
print(x)
