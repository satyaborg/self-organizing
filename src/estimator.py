import scipy
import logging
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy as np
import random
import torch
import json
import os

logger = logging.getLogger("train_log")

def normalize(x):
    # scale values between 0 - 1
    max = torch.max(torch.squeeze(x)) # nax across?
    min = torch.min(x)
    x = (x - min)/(max - min)
    return x

def binned_entropy(x, bins=100):
    x = normalize(x)
    hist, _ = np.histogram(x, bins) # discretize into bins and get the pdf
    h = scipy.stats.entropy(pk=hist, base=2)  # or shan_entropy(hist)
    return h

def get_h_binned(**kwargs):

    h_bin = [] # , h_bin2, h_bin3 = [], [], []

    # for each ae unit
    for idx, loc in enumerate(kwargs["locs"]):
        # for each hidden layer
        x = kwargs["emap"][:, :, loc[0][0]:loc[0][1], loc[1][0]:loc[1][1]]
        x = x.reshape(x.size(0), -1) # reshape (60000x10x10)-> (60000x100)
        
        # I. Binned estimator
        h_bin.append(binned_entropy(x))
        # h_bin2.append(bin_calc_information(x.numpy(), 0.5))
        # h_bin3.append(calculate_entropy(x).item())
        
    results = {'h_bin' : h_bin} #, 'h_bin2' : h_bin2, 'h_bin3' : h_bin3}
    fname = os.path.join(kwargs["path"], kwargs["prefix"]  + '-' + 'bin' + '.json')
    # fname = str(units) + '-' + str(arch) + '-' + str(run) + '-' + 'bin' + '.json'
    with open(fname, 'w') as f:
        json.dump(results, f)
    print('==> binned entropy estimates saved')