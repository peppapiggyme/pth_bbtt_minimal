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
print(f"device: {device}")

# Local
import nn_utils

nn_utils.seed_everything(42)

from nn_inputs import *
from nn_models import *

BASE_PATH = "/scratchfs/atlas/bowenzhang/ML/ntuple_v2"


def train():

    # -------------------------------------------------------------------------

    # Ntuples
    mapSigs = {
        # "NonRes_1p0" : f"{BASE_PATH}/NonRes_1p0.root",
        "NonRes_10p0" : f"{BASE_PATH}/NonRes_10p0.root",
    }

    mapMCBkgs = {
        "TTbar" : f"{BASE_PATH}/TTbar.root", 
        "SingleTop" : f"{BASE_PATH}/SingleTop.root", 
        "Zjets" : f"{BASE_PATH}/Zjets.root", 
        "Wjets" : f"{BASE_PATH}/Wjets.root", 
        "Diboson" : f"{BASE_PATH}/Diboson.root", 
        "ttV" : f"{BASE_PATH}/ttV.root", 
        "ttH" : f"{BASE_PATH}/ttH.root", 
        "VH" : f"{BASE_PATH}/VH.root", 
        "Htautau" : f"{BASE_PATH}/Htautau.root", 
    }

    mapFakeBkgs = {
        "Fake" : f"{BASE_PATH}/Fake.root", 
    }

    # hyper-parameters

    NEPOCH = 200
    BATCH_SIZE = 128

    ConfigOdd = Config_DNN_Odd
    ConfigEven = Config_DNN_Even

    # Odd or Even for *Training*
    lNtuplesOdd = list()
    lNtuplesEven = list()
    lNtuplesTrain = lNtuplesEven  # or Odd
    lNtuplesTest = lNtuplesOdd  # or Even

    # Which NN
    NeuralNetwork = BBTT_DNN

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

    # Train validation splitting
    cArrays = InputArrays(lNtuplesTrain).class_balance_basic()
    nSize = len(cArrays.weight_vec())
    fSplitVal = 0.2
    nSplitVal = int(np.floor((1 - fSplitVal) * nSize))
    lIndices = list(range(nSize))
    np.random.shuffle(lIndices)
    lIndicesTrain, lIndicesVal = lIndices[:nSplitVal], lIndices[nSplitVal:]
    
    # PyTorch DataLoaders for train / validation / test datasets
    cSamplerTrain = SubsetRandomSampler(lIndicesTrain)
    cSamplerVal = SubsetRandomSampler(lIndicesVal)
    cDataset = InputDataset(cArrays)
    cLoaderTrain = DataLoader(cDataset, batch_size=BATCH_SIZE, sampler=cSamplerTrain)
    cLoaderVal = DataLoader(cDataset, batch_size=BATCH_SIZE, sampler=cSamplerVal)

    cArraysTest = InputArrays(lNtuplesTest).class_balance_basic()
    cDatasetTest = InputDataset(cArraysTest)
    cLoaderTest = DataLoader(cDatasetTest, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Background sum of weights: {cArraysTest.weight_vec()[cArraysTest.target_vec()==0].sum()}")
    print(f"Signal     sum of weights: {cArraysTest.weight_vec()[cArraysTest.target_vec()==1].sum()}")

    print(f"Example input: {cArrays.feature_vec()[0]}")
    print(f"Example helper variables: {cArrays.other_vec()[0]}")

    # NN Model
    cNet = NeuralNetwork().to(device)
    print(cNet)
    params = list(cNet.parameters())
    # print(len(params))
    # print([p.size() for p in params])
    print("n Parameters to train: ", sum(p.numel() for p in cNet.parameters() if p.requires_grad))

    # 
    lValLosses = list()
    fMinValLoss = 1e9
    iMinValLoss = 10000
    nPatience = 10
    nPatienceCount = 5 # early stop

    cCriterion = nn.NLLLoss(reduction='none')
    fLearningRate = 0.001
    cOptimizer = optim.Adam(cNet.parameters(), lr=fLearningRate)
    cScheduler = optim.lr_scheduler.CosineAnnealingLR(cOptimizer, nPatience<<1)

    # Training
    for epoch in range(NEPOCH):  # loop over the dataset multiple times
        print(f"Epoch [{epoch}]")
        
        loss_train = 0.0
        acc_train = 0.0
        tot_train = 0.0
        
        loss_val = 0.0
        acc_val_BDT = 0.0
        acc_val_PNN = 0.0
        acc_val = 0.0
        tot_val = 0.0

        print("Train")
        for i, (x, y, w, other) in enumerate(cLoaderTrain):
            cOptimizer.zero_grad()

            cScores = cNet(x)
            vPred = cScores.argmax(dim=-1)
            vLoss = cCriterion(cScores, y)
            vLoss = vLoss.mul(w)  # weighted loss
            vLoss.mean().backward()
            cOptimizer.step()
            cScheduler.step()

            loss_train += vLoss.sum().item()
            acc_train += (vPred == y).float().mul(w).sum()
            tot_train += w.sum()

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

        print("> Loss = Tr %.6f Va %.6f" % (100 * loss_train / tot_train, 100 * loss_val / tot_val),
              ", Acc = Tr %.6f Va %.6f" % (100 * acc_train / tot_train, 100 * acc_val / tot_val),
              ", Acc BDT = Va %.6f" % (100 * acc_val_BDT / tot_val),
              ", Acc PNN500 = Va %.6f" % (100 * acc_val_PNN / tot_val))

        if loss_val < fMinValLoss:
            fMinValLoss = loss_val
            iMinValLoss = len(lValLosses)
            nPatienceCount = nPatience
        else:
            nPatienceCount -= 1

        lValLosses.append(loss_val)

        print ("> Current LR = %.6f, Minimal loss at epoch %d" % (nn_utils.get_lr(cOptimizer), iMinValLoss))
        torch.save(cNet.state_dict(), f"output/model-{epoch}.pt")

        if nPatienceCount == 0:
            print ("Returning...")
            break

    print('Finished Training')

    with torch.no_grad():
        print("Test")
        
        # load the best model
        cNet.load_state_dict(torch.load(f"output/model-{iMinValLoss}.pt"))

        loss = 0.0
        acc_BDT = 0.0
        acc_PNN = 0.0
        acc = 0.0
        tot = 0.0

        for i, (x, y, w, other) in enumerate(cLoaderTest):
            vScores = cNet(x)
            vPred = vScores.argmax(dim=-1)
            vLoss = cCriterion(vScores, y)
            vLoss = vLoss.mul(w)

            vPred_BDT = torch.round(other[:,0]*0.5 + 0.5).long()
            vPred_PNN = torch.round(other[:,1]).long()

            loss += vLoss.sum().item() 
            acc += (vPred == y).float().mul(w).sum()
            acc_BDT += (vPred_BDT == y).float().mul(w).sum()
            acc_PNN += (vPred_PNN == y).float().mul(w).sum()
            tot += w.sum()

        print("> Loss = Test %.6f" % (100 * loss / tot), 
              ", Acc = Test %.6f" % (100 * acc / tot),
              ", Acc BDT = Test %.6f" % (100 * acc_BDT / tot), 
              ", Acc PNN500 = Test %.6f" % (100 * acc_PNN / tot))

    # Plotting ROC
    with torch.no_grad():

        # load the best model
        cNet.load_state_dict(torch.load(f"output/model-{iMinValLoss}.pt"))
        
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
