# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Local
import nn_utils

nn_utils.seed_everything(42)


# Simplest toy DNN model
class BBTT_DNN(nn.Module):
    def __init__(self, nIn=5, nHidden=[64,64,64]):
        super().__init__()
        self.hidden_0 = nn.Linear(nIn, nHidden[0])
        self.hidden_1 = nn.Linear(nHidden[0], nHidden[1])
        self.hidden_2 = nn.Linear(nHidden[1], nHidden[2])
        self.hidden_3 = nn.Linear(nHidden[2], 2)
    
    def forward(self, x):
        x = torch.relu(self.hidden_0(x))
        x = torch.relu(self.hidden_1(x))
        x = torch.relu(self.hidden_2(x))
        x = F.log_softmax(self.hidden_3(x), dim=1)
        return x


def unit_test():
    dnn = BBTT_DNN()

    nParamDNN = sum(p.numel() for p in dnn.parameters() if p.requires_grad)

    nBatchSize = 10
    x1 = torch.randn(nBatchSize, 5)

    y1 = dnn(x1)

    print(y1)

    print("DNN", nParamDNN)

# unit_test()
