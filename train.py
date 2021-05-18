"""
o------o
| NOTE |
o------o

* 

o------o
| TODO |
o------o

* Class balance: reweight based on ?, now just scale factors [5e-2, 100] -> O(nSig)


"""

# Standard libraries
import os
import json
import math
import numpy as np
import time
from itertools import cycle
import collections

# Data analysis libraries
from sklearn.metrics import roc_curve, auc

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

    # -------------------------------------------------------------------------

    # Ntuples
    mapSigs = {
        "NonRes_1p0" : "/scratchfs/atlas/bowenzhang/ML/ntuple/NonRes_1p0.root",
    }

    mapMCBkgs = {
        # "NonRes_10p0" : "/scratchfs/atlas/bowenzhang/ML/ntuple/NonRes_10p0.root",
        "TTbar" : "/scratchfs/atlas/bowenzhang/ML/ntuple/TTbar.root", 
        "SingleTop" : "/scratchfs/atlas/bowenzhang/ML/ntuple/SingleTop.root", 
        "Zjets" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Zjets.root", 
        "Wjets" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Wjets.root", 
        "Diboson" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Diboson.root", 
        "ttV" : "/scratchfs/atlas/bowenzhang/ML/ntuple/ttV.root", 
        "ttH" : "/scratchfs/atlas/bowenzhang/ML/ntuple/ttH.root", 
        "VH" : "/scratchfs/atlas/bowenzhang/ML/ntuple/VH.root", 
        "Htautau" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Htautau.root", 
    }

    mapFakeBkgs = {
        "Fake" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Fake.root", 
    }

    # Magic variables

    NEPOCH = 50
    BATCH_SIZE = 256

    ConfigOdd = Config_GNN_Odd
    ConfigEven = Config_GNN_Even

    # Odd or Even for *Training*
    lNtuplesOdd = list()
    lNtuplesEven = list()
    lNtuples = lNtuplesOdd

    # Which NN
    NeuralNetwork = BBTT_GATNN

    # -------------------------------------------------------------------------

    # Odd <-> Even splitting, can be extended to multiple folds

    for sTreeName, sFileName in mapSigs.items():
        lNtuplesOdd.append(Ntuple(sFileName, sTreeName, ConfigOdd(CategorySB.SIG)))
    for sTreeName, sFileName in mapMCBkgs.items():
        lNtuplesOdd.append(Ntuple(sFileName, sTreeName, ConfigOdd(CategorySB.BKG_MC)))
    for sTreeName, sFileName in mapFakeBkgs.items():
        lNtuplesOdd.append(Ntuple(sFileName, sTreeName, ConfigOdd(CategorySB.BKG_FAKE)))

    for sTreeName, sFileName in mapSigs.items():
        lNtuplesEven.append(Ntuple(sFileName, sTreeName, ConfigEven(CategorySB.SIG)))
    for sTreeName, sFileName in mapMCBkgs.items():
        lNtuplesEven.append(Ntuple(sFileName, sTreeName, ConfigEven(CategorySB.BKG_MC)))
    for sTreeName, sFileName in mapFakeBkgs.items():
        lNtuplesEven.append(Ntuple(sFileName, sTreeName, ConfigEven(CategorySB.BKG_FAKE)))

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
    cLoaderTrain = DataLoader(cDataset, batch_size=BATCH_SIZE, sampler=cSamplerTrain)
    cLoaderVal = DataLoader(cDataset, batch_size=BATCH_SIZE, sampler=cSamplerVal)

    cArraysTest = InputArrays(lNtuplesEven)
    cDatasetTest = InputDataset(cArraysTest)
    cLoaderTest = DataLoader(cDatasetTest, batch_size=BATCH_SIZE, shuffle=True)

    print(cArraysTest.weight_vec()[cArraysTest.target_vec()==0].sum())
    print(cArraysTest.weight_vec()[cArraysTest.target_vec()==1].sum())

    print(cArrays.feature_vec()[:2])
    print(cArraysTest.feature_vec()[:2])

    # NN Model
    cNet = NeuralNetwork().to(device)
    print(cNet)
    params = list(cNet.parameters())
    print(len(params))
    print([p.size() for p in params])

    # 
    cCriterion = nn.NLLLoss(reduction='none')
    fLearningRate = 0.01
    # cOptimizer = optim.SGD(cNet.parameters(), lr=0.1)
    # cOptimizer = optim.SGD(cNet.parameters(), lr=fLearningRate, momentum=0.9, nesterov=True)
    cOptimizer = optim.Adam(cNet.parameters(), lr=fLearningRate)

    lValLosses = list()
    fMinValLoss = 1e9
    iMinValLoss = 100
    nPatience = 4
    nPatienceCount = 4

    # Training
    for epoch in range(NEPOCH):  # loop over the dataset multiple times
        print(f"Epoch [{epoch}]")
        
        loss_train = 0.0
        acc_train = 0.0
        tot_train = 0.0
        acc_train_raw = 0.0
        tot_train_raw = 0.0
        
        loss_val = 0.0
        acc_val_BDT = 0.0
        acc_val_PNN = 0.0
        acc_val = 0.0
        tot_val = 0.0

        loss_test = 0.0
        acc_test_BDT = 0.0
        acc_test_PNN = 0.0
        acc_test = 0.0
        tot_test = 0.0
        acc_test_raw = 0.0
        acc_test_BDT_raw = 0.0
        acc_test_PNN_raw = 0.0
        tot_test_raw = 0.0
        
        print("Train")
        for i, (x, y, w, _) in enumerate(cLoaderTrain):
            cOptimizer.zero_grad()

            cScores = cNet(x)
            vPred = cScores.argmax(dim=-1)
            vLoss = cCriterion(cScores, y)
            vLoss = vLoss.mul(w)
            vLoss.mean().backward()
            cOptimizer.step()

            loss_train += vLoss.sum().item()
            acc_train += (vPred == y).float().mul(w).sum()
            tot_train += w.sum()
            acc_train_raw += (vPred == y).sum().float()
            tot_train_raw += y.size()[0]
        
        print("Validation")
        with torch.no_grad():
            for i, (x, y, w, other) in enumerate(cLoaderVal):
                cScores = cNet(x)
                vPred = cScores.argmax(dim=-1)
                vLoss = cCriterion(cScores, y)
                vLoss = vLoss.mul(w)

                vPred_BDT = torch.round(other[:,0]*0.5 + 0.5).long()
                vPred_PNN = torch.round(other[:,1]).long()

                loss_val += vLoss.sum().item() 
                acc_val += (vPred == y).float().mul(w).sum()
                acc_val_BDT += (vPred_BDT == y).float().mul(w).sum()
                acc_val_PNN += (vPred_PNN == y).float().mul(w).sum()
                tot_val += w.sum()

            print("Test")
            for i, (x, y, w, other) in enumerate(cLoaderTest):
                vScores = cNet(x)
                vPred = vScores.argmax(dim=-1)
                vLoss = cCriterion(vScores, y)
                vLoss = vLoss.mul(w)

                vPred_BDT = torch.round(other[:,0]*0.5 + 0.5).long()
                vPred_PNN = torch.round(other[:,1]).long()

                loss_test += vLoss.sum().item() 
                acc_test += (vPred == y).float().mul(w).sum()
                acc_test_BDT += (vPred_BDT == y).float().mul(w).sum()
                acc_test_PNN += (vPred_PNN == y).float().mul(w).sum()
                tot_test += w.sum()
                acc_test_raw += (vPred == y).sum().float()
                acc_test_BDT_raw += (vPred_BDT == y).sum().float()
                acc_test_PNN_raw += (vPred_PNN == y).sum().float()
                tot_test_raw += y.size()[0]

        print("> Loss = Tr %.6f Va %.6f" % (100*loss_train/tot_train, 100*loss_val/tot_val),
              ", Acc = Tr %.6f Va %.6f" % (100*acc_train/tot_train, 100*acc_val/tot_val),
              ", Acc BDT = Va %.6f" % (100*acc_val_BDT/tot_val),
              ", Acc PNN = Va %.6f" % (100*acc_val_PNN/tot_val))

        print("> Loss = Tr %.6f Te %.6f" % (100*loss_train/tot_train, 100*loss_test/tot_test),
              ", Acc = Tr %.6f Te %.6f" % (100*acc_train/tot_train, 100*acc_test/tot_test),
              ", Acc BDT = Te %.6f" % (100*acc_test_BDT/tot_test),
              ", Acc PNN = Te %.6f" % (100*acc_test_PNN/tot_test))

        print("> Loss = Tr %.6f Te %.6f" % (100*loss_train/tot_train, 100*loss_test/tot_test),
              ", Acc RAW = Tr %.6f Te %.6f" % (100*acc_train_raw/tot_train_raw, 100*acc_test_raw/tot_test_raw),
              ", Acc BDT RAW = Te %.6f" % (100*acc_test_BDT_raw/tot_test_raw),
              ", Acc PNN RAW = Te %.6f" % (100*acc_test_PNN_raw/tot_test_raw))

        if loss_val < fMinValLoss:
            fMinValLoss = loss_val
            iMinValLoss = len(lValLosses)
        else:
            nPatienceCount -= 1

        if nPatienceCount == 0:
            fLearningRate = fLearningRate * 0.8
            nPatienceCount = nPatience

        lValLosses.append(loss_val)

        print ("> Current LR = %.6f, Minimal loss at epoch %d" % (fLearningRate, iMinValLoss))

    print('Finished Training')

    # Plotting ROC
    with torch.no_grad():
        cScores = cNet(cArraysTest.feature_vec())
        cScoresSignal = cScores[:,1]
        y = cArraysTest.target_vec()
        w = cArraysTest.weight_vec()

        sTag = "DNNvsBDT"
        mapCurves = collections.OrderedDict()
        mapCurves["[new] DNN"] = roc_curve(y, cScoresSignal.numpy(), sample_weight=w)
        mapCurves["SMHH BDT"] = roc_curve(y, cArraysTest.other_vec()[:,0], sample_weight=w)
        mapCurves["X500 PNN"] = roc_curve(y, cArraysTest.other_vec()[:,1], sample_weight=w)

        nn_utils.plotROCs(sTag, mapCurves)

train()
