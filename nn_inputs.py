"""
o------o
| NOTE |
o------o

* Use uproot to retrieve numpy arrays from ROOT flat ntuples

o------o
| TODO |
o------o

* No RAM optimisation, hope there's magic behind numpy, very likely wasting memories


"""

# Standard libraries
import os
import json
import math
import numpy as np
import time

# PyTorch
import torch
import torch.nn.functional as F
# device = torch.device("cuda:0") if torch.cuda.is_available else torch.cuda("cpu")
device = torch.device("cpu")
from torch.utils.data import Dataset


# Local
import uproot
import nn_utils


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class CategorySB(object):
    SIG = 0
    BKG_MC = 1
    BKG_FAKE = 2


# -----------------------------------------------------------------------------
# Configs -> tell ntuple tool what to do
# -----------------------------------------------------------------------------

class Config(object):
    def __init__(self, category):
        """
        NOTE the last of weights will be used as default sample weight
        See Config._getArrays()
        """ 
        self.category = category
        self.features = list()
        self.weights = list()
        self.selVars = list()
    
    def sel_vec(self, mapSelVars):
        """
        return a numpy array of conditions
        [True, True, False, ..., False, False]
        """
        raise NotImplementedError

    def features_vec(self, mapFeatures):
        """
        return a numpy array of features
        reshaping and concatenate implemented here
        """
        raise NotImplementedError

    def target_vec(self, mapFeatures):
        """
        return a numpy array of targets
        for signals, typically np.ones
        for backgrounds, typically np.zeros
        """
        raise NotImplementedError
        
    def other_vec(self, mapOthers):
        for sOther in self.others:
            mapOthers[sOther] = mapOthers[sOther].reshape((-1, 1))
        return np.concatenate([mapOthers[f] for f in self.others], axis=1)

    def trans(self, mapFeatures):
        """
        updates the content in mapFeatures based 
        on the tranformation defined here
        """
        raise NotImplementedError
    
    def features_vec_basic(self, mapFeatures):
        for sFeature in self.features:
            mapFeatures[sFeature] = mapFeatures[sFeature].reshape((-1, 1))
        return np.concatenate([mapFeatures[f] for f in self.features], axis=1)

    def trans_standarize(self, mapFeatures, use_cache=True):
        for sFeature in self.features:
            if use_cache:
                fOffSet, fScale = nn_utils.cached_trans_standarise()[sFeature]
            else:
                raise RuntimeError("I think it's wrong if cache is not used!")
                # fOffSet, fScale = mapFeatures[sFeature].mean(), 0.5 / mapFeatures[sFeature].std()
            # print(sFeature, fOffSet, fScale)
            mapFeatures[sFeature] = np.multiply(np.subtract(mapFeatures[sFeature], fOffSet), fScale)
    
    def class_balance(self, vWeight, vTarget, *targs, **kwargs):
        raise NotImplementedError

    def class_balance_basic(self, vWeights, vTarget, w1, w2):
        vWeights[vTarget==0] = vWeights[vTarget==0] * w1
        vWeights[vTarget==1] = vWeights[vTarget==1] * w2


# -----------------------------------------------------------------------------
# Config for DNN
# -----------------------------------------------------------------------------

class Config_DNN(Config):
    def __init__(self, category):
        super().__init__(category)
        self.features = [b"mHH", b"mMMC", b"mBB", b"dRTauTau", b"dRBB"]
        self.weights = [b"weight"]
        self.selVars = [b"is_sr", b"is_fake_cr", b"event_number"]
        self.others = [b"SMBDT", b"PNN500"]
    
    def sel_vec(self, mapSelVars):
        if self.category == CategorySB.BKG_FAKE:
            return mapSelVars[b"is_fake_cr"]
        return mapSelVars[b"is_sr"]

    def features_vec(self, mapFeatures):
        return self.features_vec_basic(mapFeatures)

    def target_vec(self, mapFeatures):
        if self.category == CategorySB.SIG:
            return np.ones(mapFeatures[self.features[-1]].shape[0], dtype=np.long)
        elif self.category == CategorySB.BKG_MC or self.category == CategorySB.BKG_FAKE:
            return np.zeros(mapFeatures[self.features[-1]].shape[0], dtype=np.long)
        else:
            raise RuntimeError(f"Category [{self.category}] not defined")

    def trans(self, mapFeatures):
        self.trans_standarize(mapFeatures, True)

    def class_balance(self, vWeights, vTarget, *targs, **kwargs):
        self.class_balance_basic(vWeight, vTarget, targs[0], targs[1])


# Odd fold
class Config_DNN_Odd(Config_DNN):
    def __init__(self, category):
        super().__init__(category)
    
    def sel_vec(self, mapSelVars):
        return np.logical_and(super().sel_vec(mapSelVars), \
            np.equal(np.mod(mapSelVars[b"event_number"], 2), 1))


# Even fold
class Config_DNN_Even(Config_DNN):
    def __init__(self, category):
        super().__init__(category)
    
    def sel_vec(self, mapSelVars):
        return np.logical_and(super().sel_vec(mapSelVars), \
            np.equal(np.mod(mapSelVars[b"event_number"], 2), 0))


# -----------------------------------------------------------------------------
# Config for GNN
# -----------------------------------------------------------------------------

class Config_GNN(Config):
    def __init__(self, category):
        """
        The order matters!
        """
        super().__init__(category)
        self.features = [
            b"mmc_pt", b"mmc_m", b"mmc_eta", b"mmc_phi", 
            b"bb_corr_pt", b"bb_corr_m", b"bb_corr_eta", b"bb_corr_phi", 
            b"tau0_pt", b"tau0_m", b"tau0_eta", b"tau0_phi", 
            b"tau1_pt", b"tau1_m", b"tau1_eta", b"tau1_phi", 
            b"b0_pt", b"b0_m", b"b0_eta", b"b0_phi", 
            b"b1_pt", b"b1_m", b"b1_eta", b"b1_phi", 
        ]
        self.weights = [b"weight"]
        self.selVars = [b"is_sr", b"is_fake_cr", b"event_number"]
        self.others = [b"SMBDT", B"PNN500"]
    
    def sel_vec(self, mapSelVars):
        if self.category == CategorySB.BKG_FAKE:
            return mapSelVars[b"is_fake_cr"]
        return mapSelVars[b"is_sr"]

    def features_vec(self, mapFeatures):
        for sFeature in self.features:
            mapFeatures[sFeature] = mapFeatures[sFeature].reshape((-1, 1))
        return np.concatenate([mapFeatures[f] for f in self.features], axis=1).reshape((-1, 6, 4))

    def target_vec(self, mapFeatures):
        if self.category == CategorySB.SIG:
            return np.ones(mapFeatures[self.features[-1]].shape[0], dtype=np.long)
        elif self.category == CategorySB.BKG_MC or self.category == CategorySB.BKG_FAKE:
            return np.zeros(mapFeatures[self.features[-1]].shape[0], dtype=np.long)
        else:
            raise RuntimeError(f"Category [{self.category}] not defined")

    def trans(self, mapFeatures):
        for sFeature in self.features:
            if sFeature.endswith(b"_pt"):
                mapFeatures[sFeature] = np.log10(mapFeatures[sFeature])
        self.trans_standarize(mapFeatures, True)

    def class_balance(self, vWeights, vTarget, *targs, **kwargs):
        self.class_balance_basic(vWeights, vTarget, targs[0], targs[1])

# Odd fold
class Config_GNN_Odd(Config_GNN):
    def __init__(self, category):
        super().__init__(category)
    
    def sel_vec(self, mapSelVars):
        return np.logical_and(super().sel_vec(mapSelVars), \
            np.equal(np.mod(mapSelVars[b"event_number"], 2), 1))


# Even fold
class Config_GNN_Even(Config_GNN):
    def __init__(self, category):
        super().__init__(category)
    
    def sel_vec(self, mapSelVars):
        return np.logical_and(super().sel_vec(mapSelVars), \
            np.equal(np.mod(mapSelVars[b"event_number"], 2), 0))


# -----------------------------------------------------------------------------
# Ntuple wrapper
# -----------------------------------------------------------------------------

class Ntuple(object):
    def __init__(self, sFileName, sTreeName, cConfig):
        self.cConfig = cConfig
        self.cData = uproot.open(sFileName)[sTreeName]
        self.mapFeatures = dict()
        self.mapWeights = dict()
        self.mapSelVars = dict()
        self.mapOthers = dict()
        self.vFeatures = None
        self.vTarget = None
        self.vWeights = None
        self.vOthers = None
        self.vSel = None

        self._initialize()  # arrays are read from file in this steps

    def _initialize(self):
        self.mapFeatures = self.cData.arrays(self.cConfig.features)
        self.mapWeights = self.cData.arrays(self.cConfig.weights)
        self.mapSelVars = self.cData.arrays(self.cConfig.selVars)
        self.mapOthers = self.cData.arrays(self.cConfig.others)
        self.vSel = self.cConfig.sel_vec(self.mapSelVars)

        self._applySelection()
        self._applyTranformation()
        self._getArrays()

    def _applySelection(self):
        for sFeature in self.cConfig.features:
            self.mapFeatures[sFeature] = self.mapFeatures[sFeature][self.vSel]
        for sWeight in self.cConfig.weights:
            self.mapWeights[sWeight] = self.mapWeights[sWeight][self.vSel]
        for sOther in self.cConfig.others:
            self.mapOthers[sOther] = self.mapOthers[sOther][self.vSel]

    def _applyTranformation(self):
        """
        NOTE The transformation can be different between ntuples
        when the trans is dependent on the feature vectors

        NOTE Max flexibility, Max care!

        TODO Confirm if the flexibility is useful here
        Otherwise let's do the other way around, 
        i.e. merge first, than do the transformations, vector constructions, ...

        Take care, especially for the standarized one
        Don't assume they are the same for all
        For consistency, use the cached version 
        
        For Log, Tanh trans, they are not dependent on the feature vectors
        """
        self.cConfig.trans(self.mapFeatures)
    
    def _getArrays(self):
        """
        Need to specify the name
        Feature Vector, Target Vector, Weight Vector
        """
        self.vFeatures = self.cConfig.features_vec(self.mapFeatures)
        self.vTarget = self.cConfig.target_vec(self.mapFeatures)
        self.vWeight = self.mapWeights[self.cConfig.weights[-1]]
        self.vOthers = self.cConfig.other_vec(self.mapOthers)
        self.cConfig.class_balance(self.vWeight, self.vTarget, 0.025, 20)

    def feature_vec(self):
        return self.vFeatures
    
    def target_vec(self):
        return self.vTarget

    def weight_vec(self):
        return self.vWeight

    def other_vec(self):
        return self.vOthers


# -----------------------------------------------------------------------------
# InputArrays -> Merge arrays from ntuples and port to torch.Tensor
# -----------------------------------------------------------------------------

class InputArrays(object):
    def __init__(self, lNtuples):
        self._fWeightScale = 100
        self._lNtuples = lNtuples
        self._f_vec = torch.Tensor(self._merged("feature_vec")).to(device)
        self._t_vec = torch.LongTensor(self._merged("target_vec")).to(device)
        self._w_vec = torch.Tensor(self._merged("weight_vec")).to(device).mul(self._fWeightScale)
        self._o_vec = torch.Tensor(self._merged("other_vec")).to(device)

    def _merged(self, func_vec):
        return np.concatenate([getattr(ntup, func_vec)() for ntup in self._lNtuples], axis=0)

    def feature_vec(self):
        return self._f_vec

    def target_vec(self):
        return self._t_vec

    def weight_vec(self):
        return self._w_vec

    def other_vec(self):
        return self._o_vec


# -----------------------------------------------------------------------------
# InputDataset -> Inherite from PyTorch Dataset, Used by DataLoader
# -----------------------------------------------------------------------------

class InputDataset(Dataset):
    def __init__(self, cInputArray):
        super().__init__()
        self.cInputArray = cInputArray
        
    def __len__(self):
        return self.cInputArray.target_vec().size()[0]
        
    def __getitem__(self, index):
        return self.cInputArray.feature_vec()[index], \
               self.cInputArray.target_vec()[index], \
               self.cInputArray.weight_vec()[index], \
               self.cInputArray.other_vec()[index]


def unit_test():
    mapSigs = {
        "NonRes_1p0" : "/scratchfs/atlas/bowenzhang/ML/ntuple/NonRes_1p0.root",
    }

    mapBkgs = {
        "TTbar" : "/scratchfs/atlas/bowenzhang/ML/ntuple/TTbar.root", 
        "Zjets" : "/scratchfs/atlas/bowenzhang/ML/ntuple/Zjets.root", 
    }

    lNtuples = list()

    for sTreeName, sFileName in mapSigs.items():
        lNtuples.append(Ntuple(sFileName, sTreeName, Config_DNN_Odd(CategorySB.SIG)))

    for sTreeName, sFileName in mapBkgs.items():
        lNtuples.append(Ntuple(sFileName, sTreeName, Config_DNN_Odd(CategorySB.BKG_MC)))

    ia = InputArrays(lNtuples)

    print(ia.feature_vec())
    print(ia.target_vec())
    print(ia.weight_vec())
    print(ia.other_vec())

    print(ia.feature_vec().shape)
    print(ia.target_vec().shape)
    print(ia.weight_vec().shape)
    print(ia.other_vec().shape)

