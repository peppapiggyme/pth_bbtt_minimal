# Standard libraries
import os
import json
import math
import numpy as np
import time

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# device = torch.device("cuda:0") if torch.cuda.is_available else torch.cuda("cpu")
device = torch.device("cpu")
print(device)

# Local
import nn_utils

nn_utils.seed_everything(42)

import nn_inputs
import nn_models

def train():
    # Ntuples
    mapSigs = {
        "NonRes_1p0" : "/scratchfs/atlas/bowenzhang/ML/ntuple/NonRes_1p0.root",
    }

    mapBkgs = {
        # "NonRes_10p0" : "/scratchfs/atlas/bowenzhang/ML/ntuple/NonRes_10p0.root",
        "TTbar" : "/scratchfs/atlas/bowenzhang/ML/ntuple/TTbar.root", 
        "Zjets" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Zjets.root", 
        "Diboson" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Diboson.root", 
        "Fake" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Fake.root", 
        "Htautau" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Htautau.root", 
        "SingleTop" : "/scratchfs/atlas/bowenzhang/ML/ntuple/SingleTop.root", 
        "ttH" : "/scratchfs/atlas/bowenzhang/ML/ntuple/ttH.root", 
        "VH" : "/scratchfs/atlas/bowenzhang/ML/ntuple/VH.root", 
        "Wjets" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Wjets.root", 
    }

    lNtuplesOdd = list()
    lNtuplesEven = list()

    for sTreeName, sFileName in mapSigs.items():
        lNtuplesOdd.append(
            nn_inputs.Ntuple(sFileName, sTreeName, nn_inputs.Config_DNN_Odd(nn_inputs.CategorySB.SIG)))
    for sTreeName, sFileName in mapBkgs.items():
        lNtuplesOdd.append(
            nn_inputs.Ntuple(sFileName, sTreeName, nn_inputs.Config_DNN_Odd(nn_inputs.CategorySB.BKG)))

    for sTreeName, sFileName in mapSigs.items():
        lNtuplesEven.append(
            nn_inputs.Ntuple(sFileName, sTreeName, nn_inputs.Config_DNN_Even(nn_inputs.CategorySB.SIG)))
    for sTreeName, sFileName in mapBkgs.items():
        lNtuplesEven.append(
            nn_inputs.Ntuple(sFileName, sTreeName, nn_inputs.Config_DNN_Even(nn_inputs.CategorySB.BKG)))

    # Odd or Even
    lNtuples = lNtuplesOdd

    # PyTorch DataLoaders
    cArrays = nn_inputs.InputArrays(lNtuples)
    nSize = len(cArrays.weight_vec())
    fSplitVal = 0.2
    nSplitVal = int(np.floor((1 - fSplitVal) * nSize))
    lIndices = list(range(nSize))
    np.random.shuffle(lIndices)
    lIndicesTrain, lIndicesVal = lIndices[:nSplitVal], lIndices[nSplitVal:]
    
    cSamplerTrain = SubsetRandomSampler(lIndicesTrain)
    cSamplerVal = SubsetRandomSampler(lIndicesVal)
    cDataset = nn_inputs.InputDataset(cArrays)
    cLoaderTrain = DataLoader(cDataset, batch_size=64, sampler=cSamplerTrain)
    cLoaderVal = DataLoader(cDataset, batch_size=64, sampler=cSamplerVal)

    cArraysTest = nn_inputs.InputArrays(lNtuplesEven)
    cDatasetTest = nn_inputs.InputDataset(cArraysTest)
    cLoaderTest = DataLoader(cDatasetTest, batch_size=64, shuffle=True)

    print(cArrays.feature_vec()[:2])
    print(cArraysTest.feature_vec()[:2])

    # NN Model
    net = nn_models.BBTT_DNN().to(device)
    print(net)
    params = list(net.parameters())
    print(len(params))
    print([p.size() for p in params])

    # 
    criterion = nn.NLLLoss(weight=torch.Tensor([1, 200]), reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)

    # Training
    for epoch in range(100):  # loop over the dataset multiple times
        print(f"Epoch [{epoch}]")
        
        loss_train = 0.0
        acc_train = 0.0
        tot_train = 0.0
        acc_train_raw = 0.0
        tot_train_raw = 0.0
        
        loss_val = 0.0
        acc_val_bdt = 0.0
        acc_val_pnn = 0.0
        acc_val = 0.0
        tot_val = 0.0

        loss_test = 0.0
        acc_test_bdt = 0.0
        acc_test_pnn = 0.0
        acc_test = 0.0
        tot_test = 0.0
        acc_test_raw = 0.0
        acc_test_bdt_raw = 0.0
        acc_test_pnn_raw = 0.0
        tot_test_raw = 0.0
        
        for i, (x, y, w, _) in enumerate(cLoaderTrain):
            optimizer.zero_grad()

            scores = net(x)
            y_pred = scores.argmax(dim=-1)
            loss = criterion(scores, y)
            loss = loss.mul(w)
            loss.mean().backward()
            optimizer.step()

            loss_train += loss.sum().item()
            acc_train += (y_pred == y).float().mul(w).sum()
            tot_train += w.sum()
            acc_train_raw += (y_pred == y).sum().float()
            tot_train_raw += y.size()[0]
        
        with torch.no_grad():
            for i, (x, y, w, other) in enumerate(cLoaderVal):
                scores = net(x)
                y_pred = scores.argmax(dim=-1)
                loss = criterion(scores, y)
                loss = loss.mul(w)

                y_pred_bdt = torch.round(other[:,0]*0.5 + 0.5).long()
                y_pred_pnn = torch.round(other[:,1]).long()

                loss_val += loss.sum().item() 
                acc_val += (y_pred == y).float().mul(w).sum()
                acc_val_bdt += (y_pred_bdt == y).float().mul(w).sum()
                acc_val_pnn += (y_pred_pnn == y).float().mul(w).sum()
                tot_val += w.sum()

            for i, (x, y, w, other) in enumerate(cLoaderTest):
                scores = net(x)
                y_pred = scores.argmax(dim=-1)
                loss = criterion(scores, y)
                loss = loss.mul(w)

                y_pred_bdt = torch.round(other[:,0]*0.5 + 0.5).long()
                y_pred_pnn = torch.round(other[:,1]).long()

                loss_test += loss.sum().item() 
                acc_test += (y_pred == y).float().mul(w).sum()
                acc_test_bdt += (y_pred_bdt == y).float().mul(w).sum()
                acc_test_pnn += (y_pred_pnn == y).float().mul(w).sum()
                tot_test += w.sum()
                acc_test_raw += (y_pred == y).sum().float()
                acc_test_bdt_raw += (y_pred_bdt == y).sum().float()
                acc_test_pnn_raw += (y_pred_pnn == y).sum().float()
                tot_test_raw += y.size()[0]

        print("> Loss = Tr %.6f Va %.6f" % (100*loss_train/tot_train, 100*loss_val/tot_val),
              ", Acc = Tr %.6f Va %.6f" % (100*acc_train/tot_train, 100*acc_val/tot_val),
              ", Acc BDT = Va %.6f" % (100*acc_val_bdt/tot_val),
              ", Acc PNN = Va %.6f" % (100*acc_val_pnn/tot_val))

        print("> Loss = Tr %.6f Te %.6f" % (100*loss_train/tot_train, 100*loss_test/tot_test),
              ", Acc = Tr %.6f Te %.6f" % (100*acc_train/tot_train, 100*acc_test/tot_test),
              ", Acc BDT = Te %.6f" % (100*acc_test_bdt/tot_test),
              ", Acc PNN = Te %.6f" % (100*acc_test_pnn/tot_test))

        print("> Loss = Tr %.6f Te %.6f" % (100*loss_train/tot_train, 100*loss_test/tot_test),
              ", Acc RAW = Tr %.6f Te %.6f" % (100*acc_train_raw/tot_train_raw, 100*acc_test_raw/tot_test_raw),
              ", Acc BDT RAW = Te %.6f" % (100*acc_test_bdt_raw/tot_test_raw),
              ", Acc PNN RAW = Te %.6f" % (100*acc_test_pnn_raw/tot_test_raw))

    print('Finished Training')


train()
