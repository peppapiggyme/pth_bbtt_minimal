# Standard libraries
import os
import json
import math
import numpy as np
import time
from itertools import cycle

# Data analysis libraries
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

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

from nn_inputs import *
from nn_models import *

def train():
    # Ntuples
    mapSigs = {
        "NonRes_1p0" : "/scratchfs/atlas/bowenzhang/ML/ntuple/NonRes_1p0.root",
    }

    mapMCBkgs = {
        # "NonRes_10p0" : "/scratchfs/atlas/bowenzhang/ML/ntuple/NonRes_10p0.root",
        "TTbar" : "/scratchfs/atlas/bowenzhang/ML/ntuple/TTbar.root", 
        "Zjets" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Zjets.root", 
        "Diboson" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Diboson.root", 
        "Htautau" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Htautau.root", 
        "SingleTop" : "/scratchfs/atlas/bowenzhang/ML/ntuple/SingleTop.root", 
        "ttH" : "/scratchfs/atlas/bowenzhang/ML/ntuple/ttH.root", 
        "VH" : "/scratchfs/atlas/bowenzhang/ML/ntuple/VH.root", 
        "Wjets" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Wjets.root", 
    }

    mapFakeBkgs = {
        "Fake" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Fake.root", 
    }
    
    lNtuplesOdd = list()
    lNtuplesEven = list()

    for sTreeName, sFileName in mapSigs.items():
        lNtuplesOdd.append(Ntuple(sFileName, sTreeName, Config_DNN_Odd(CategorySB.SIG)))
    for sTreeName, sFileName in mapMCBkgs.items():
        lNtuplesOdd.append(Ntuple(sFileName, sTreeName, Config_DNN_Odd(CategorySB.BKG_MC)))
    for sTreeName, sFileName in mapFakeBkgs.items():
        lNtuplesOdd.append(Ntuple(sFileName, sTreeName, Config_DNN_Odd(CategorySB.BKG_FAKE)))

    for sTreeName, sFileName in mapSigs.items():
        lNtuplesEven.append(Ntuple(sFileName, sTreeName, Config_DNN_Even(CategorySB.SIG)))
    for sTreeName, sFileName in mapMCBkgs.items():
        lNtuplesEven.append(Ntuple(sFileName, sTreeName, Config_DNN_Even(CategorySB.BKG_MC)))
    for sTreeName, sFileName in mapFakeBkgs.items():
        lNtuplesEven.append(Ntuple(sFileName, sTreeName, Config_DNN_Even(CategorySB.BKG_FAKE)))

    # Odd or Even
    lNtuples = lNtuplesOdd

    # PyTorch DataLoaders
    cArrays = InputArrays(lNtuples)
    nSize = len(cArrays.weight_vec())
    fSplitVal = 0.2
    nSplitVal = int(np.floor((1 - fSplitVal) * nSize))
    lIndices = list(range(nSize))
    np.random.shuffle(lIndices)
    lIndicesTrain, lIndicesVal = lIndices[:nSplitVal], lIndices[nSplitVal:]
    
    cSamplerTrain = SubsetRandomSampler(lIndicesTrain)
    cSamplerVal = SubsetRandomSampler(lIndicesVal)
    cDataset = InputDataset(cArrays)
    cLoaderTrain = DataLoader(cDataset, batch_size=64, sampler=cSamplerTrain)
    cLoaderVal = DataLoader(cDataset, batch_size=64, sampler=cSamplerVal)

    cArraysTest = InputArrays(lNtuplesEven)
    cDatasetTest = InputDataset(cArraysTest)
    cLoaderTest = DataLoader(cDatasetTest, batch_size=64, shuffle=True)

    print(cArraysTest.weight_vec()[cArraysTest.target_vec()==0].sum())
    print(cArraysTest.weight_vec()[cArraysTest.target_vec()==1].sum())

    print(cArrays.feature_vec()[:2])
    print(cArraysTest.feature_vec()[:2])

    # NN Model
    net = BBTT_DNN().to(device)
    print(net)
    params = list(net.parameters())
    print(len(params))
    print([p.size() for p in params])

    # 
    criterion = nn.NLLLoss(weight=torch.Tensor([5e-2, 100]), reduction='none')
    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)

    # Training
    for epoch in range(1):  # loop over the dataset multiple times
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

    # Plotting ROC
    with torch.no_grad():
        scores = net(cArraysTest.feature_vec())
        signal_score = scores[:,1]
        # fpr_raw, tpr_raw, _ = roc_curve(cArraysTest.target_vec(), signal_score.numpy())
        fpr_wtd, tpr_wtd, _ = roc_curve(cArraysTest.target_vec(), signal_score.numpy(), sample_weight=cArraysTest.weight_vec())
        fpr_bdt, tpr_bdt, _ = roc_curve(cArraysTest.target_vec(), cArraysTest.other_vec()[:,0], sample_weight=cArraysTest.weight_vec())
        fpr_pnn, tpr_pnn, _ = roc_curve(cArraysTest.target_vec(), cArraysTest.other_vec()[:,1], sample_weight=cArraysTest.weight_vec())
        # TODO need a post-processing of the fpr, tpr to calculate AUC
        # roc_auc = auc(fpr_wtd, tpr_wtd)
        print(len(fpr_wtd), len(tpr_wtd))
        plt.figure(figsize=(6, 6))
        lw = 2
        # plt.plot(fpr_raw, tpr_raw, color='darkgreen', lw=lw, label='Raw (%0.2f)' % roc_auc)
        plt.plot(fpr_wtd, tpr_wtd, color='darkorange', lw=lw, label="Weighted")
        plt.plot(fpr_bdt, tpr_bdt, color='darkred', lw=lw, label="BDT")
        plt.plot(fpr_pnn, tpr_pnn, color='darkblue', lw=lw, label="PNN")
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig("../test.png")


train()
