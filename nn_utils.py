import random, os
import numpy as np
import torch

# Data analysis libraries
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score

# Imports for plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

# -----------------------------------------------------------------------------
# third library settings
# -----------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------

def cached_trans_standarise():
    """
    Define new ones here
    """

    cache = {
        b"mHH": (500000, 2.5e-6),
        b"mMMC": (125000, 2.5e-5),
        b"mBB": (125000, 2.5e-5),
        b"dRTauTau": (2, 0.5),
        b"dRBB": (2, 0.5),
        b"mmc_pt": (5, 2.5), 
        b"mmc_m": (125000, 2.5e-5), 
        b"mmc_eta": (0, 0.5), 
        b"mmc_phi": (0, 0.3), 
        b"bb_corr_pt": (5, 2.5), 
        b"bb_corr_m": (125000, 2.5e-5), 
        b"bb_corr_eta": (0, 0.5), 
        b"bb_corr_phi": (0, 0.3), 
        b"tau0_pt": (5, 2.5), 
        b"tau0_m": (0, 0), 
        b"tau0_eta": (0, 0.5), 
        b"tau0_phi": (0, 0.3), 
        b"tau1_pt": (5, 2.5), 
        b"tau1_m": (0, 0), 
        b"tau1_eta": (0, 0.5), 
        b"tau1_phi": (0, 0.3), 
        b"b0_pt": (5, 2.5), 
        b"b0_m": (12000, 5e-5), 
        b"b0_eta": (0, 0.5), 
        b"b0_phi": (0, 0.3), 
        b"b1_pt": (5, 2.5), 
        b"b1_m": (12000, 5e-5), 
        b"b1_eta": (0, 0.5), 
        b"b1_phi": (0, 0.3), 
    }

    return cache


# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------

def plotROCs(sTag, mapCurves):
    """
    tag -> name to save
    mapCurves -> a map of {labal: str, (fpr, tpr)}
    """
    # TODO need post-processing before calculating AUC -> auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    # plt.plot(vFalsePosRate_raw, vTruePosRate_raw, color='darkgreen', lw=lw, label='Raw (%0.2f)' % roc_auc)
    
    for sLabel, cCurves in mapCurves.items():
        # vFalsePosRate, vTruePosRate, vThreshold = cCurves
        vFalsePosRate, vTruePosRate, _ = cCurves
        plt.plot(vFalsePosRate, vTruePosRate, label=sLabel)

    # Baseline
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC")  # Receiver operating characteristic
    plt.legend(loc="lower right")
    plt.savefig(f"../output/roc_{sTag}.png")
