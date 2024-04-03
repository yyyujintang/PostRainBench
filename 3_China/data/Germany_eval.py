# from data.dataset import CustomTensorDataset
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
from data.transforms import ClassifyByThresholds
from einops import rearrange
from sklearn.metrics import mean_absolute_error
import pandas as pd

# thresholds = [0.2, 0.5, 1, 2, 5]
# window_sizes = [4, 7, 14, 21]
# thresholds = [0.00001, 2]
thresholds = [1, 2]

# def leps(y1,y2,climato):
#     if len(y1.shape)==0:
#         return abs(np.sum(y1>climato) - np.sum(y2>climato))/len(climato)
#     else:
#         return np.mean([leps(y1[i],y2[i],climato) for i in range(len(y1))])
    
def booleanize(array, threshold):
    booleanized = array >= threshold
    return booleanized.astype(int) 

def contingencytable(veri,perc,threshh):
    hits=np.sum((veri>=threshh)*(perc>=threshh))
    falsealarms=np.sum((veri<threshh)*(perc>=threshh))
    misses=np.sum((veri>=threshh)*(perc<threshh))
    correctnegatives=np.sum((veri<threshh)*(perc<threshh))
    return hits, falsealarms, misses, correctnegatives

def freqbias(hits,falsealarms, misses, correctnegatives):
    return (hits+falsealarms)/(hits+misses)

def ets(hits,falsealarms, misses, correctnegatives):
    hitsrandom=(hits+misses)*(hits+falsealarms)/np.sum((hits,falsealarms, misses, correctnegatives))
    return (hits-hitsrandom)/(hits+misses+falsealarms-hitsrandom)

def lor(hits,falsealarms, misses, correctnegatives):
    return np.log((hits*correctnegatives)/(misses*falsealarms))

def pod(hits,falsealarms, misses, correctnegatives):
    return hits/(hits+misses)

def far(hits,falsealarms, misses, correctnegatives):
    return falsealarms/(falsealarms+hits)

def csi(hits,falsealarms, misses, correctnegatives):
    return hits/(hits+falsealarms+misses)

def acc(hits,falsealarms, misses, correctnegatives):
    return (hits+correctnegatives)/(hits+falsealarms+misses+correctnegatives)


# save_path = "/mnt/ssd1/yujint/GermanyData/"
# tst_y = np.load(save_path + '01_tst_y.npy')
# tst_c = np.load(save_path + '01_tst_c.npy')
# cosmo = np.load(save_path + '01_tst_c.npy')
cosmo = np.load('test_nwp_rain.npy')
print('cosmo.shape =', cosmo.shape)
tst_y_lres = np.load('test_gt.npy')
print('tst_y_lres.shape =', tst_y_lres.shape)

# cosmo = torch.from_numpy(cosmo)
# tst_y_lres = torch.from_numpy(tst_y_lres)
# cosmo = torch.nn.functional.interpolate(cosmo.unsqueeze(0), size=(64,64), mode='bilinear', align_corners=False).squeeze().numpy()
# tst_y_lres = torch.nn.functional.interpolate(tst_y_lres.unsqueeze(0),  size=(64,64), mode='bilinear', align_corners=False).squeeze().numpy()
# print('cosmo.shape =', cosmo.shape)
# print('tst_y_lres.shape =', tst_y_lres.shape)

results_cosmo = {}

# results_cosmo['mae'] = mean_absolute_error(tst_y_lres.flatten(), cosmo.flatten())
# results_cosmo['leps'] = leps(tst_y_lres.flatten(), cosmo.flatten(), cli)


results_cosmo['acc'] = {}
results_cosmo['pod'] = {}
results_cosmo['csi'] = {}
results_cosmo['far'] = {}
results_cosmo['freqbias'] = {}


for t in thresholds:

    hits, falsealarms, misses, correctnegatives = contingencytable(tst_y_lres, cosmo, t)
    results_cosmo['acc'][str(t)] = acc(hits, falsealarms, misses, correctnegatives)
    results_cosmo['pod'][str(t)] = pod(hits, falsealarms, misses, correctnegatives)
    results_cosmo['csi'][str(t)] = csi(hits, falsealarms, misses, correctnegatives)
    results_cosmo['far'][str(t)] = far(hits, falsealarms, misses, correctnegatives)
    results_cosmo['freqbias'][str(t)] = freqbias(hits, falsealarms, misses, correctnegatives)


df = pd.DataFrame(results_cosmo)

# print(results_cosmo)
print(df)