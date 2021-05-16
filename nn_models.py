# Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Local
import nn_utils

nn_utils.seed_everything(42)

class BBTT_DNN(nn.Module):
    def __init__(self):
        super(BBTT_DNN, self).__init__()
        self.hidden_0 = nn.Linear(5, 32)
        self.hidden_1 = nn.Linear(32, 32)
        self.hidden_2 = nn.Linear(32, 32)
        self.hidden_3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = torch.relu(self.hidden_0(x))
        x = torch.relu(self.hidden_1(x))
        x = torch.relu(self.hidden_2(x))
        x = F.log_softmax(self.hidden_3(x), dim=1)
        return x

class BBTT_MPNN(nn.Module):
    pass


def unit_test():
    pass