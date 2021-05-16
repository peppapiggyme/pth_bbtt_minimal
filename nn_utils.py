import random, os
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
    }

    return cache