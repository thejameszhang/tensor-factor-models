"""
========================================================================================================================
Example usage: 
python3 pooled_pca_oos_multiperiod_ver3.py --dataset=char_anom --lst_window_size=120 --lst_K=1,3,5,10,15,20,25 --lst_lags=36,60,90,120 --max_horizon=36 --max_lag=120
========================================================================================================================

Author: James Zhang
Date: November 2024
"""
import argparse
import os
import pprint

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='scs', help='char_anom, scs, wrds' )
parser.add_argument('--lst_K', type=str, default='1,3,5,10,20',
    help='number of factors separated by ","' )
parser.add_argument('--lst_window_size', type=str, default='60,120,240',
    help='list of rolling window sizes separated by ","')
parser.add_argument('--max_horizon', type=int, default=36, 
    help='max horizon in multiperiod portfolio' )
parser.add_argument('--gamma', type=int, default=-1, help='gamma' )
parser.add_argument('--max_lag', type=int, default=60, help='max lag' )
parser.add_argument('--lst_lags', type=str, default='12,24,36,60')
parser.add_argument('--start', type=str, default='default')
args = parser.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

#%%
# from types import SimpleNamespace
# args = SimpleNamespace(dataset='char_anom', lst_K='1,3,5,10,20,30', 
#                        lst_window_size='60,120', max_horizon=36)

config = args.dataset
assert args.dataset in ('char_anom', 'wrds', 'scs')
os.environ['CONFIG'] = config

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
os.environ['CUDA_VISIBLE_DEVICES'] = "1" # Titan and FP64 mode
jax.config.update('jax_platform_name', 'gpu')
print("JAX is using device:", jax.devices()[0], jax.devices())

import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from typing import List
from tfm.utils._load_configs import *
from tfm.utils._eval import *
from tfm.models.pca import Pooled_PCA_Multiperiod_Unmapped

# Further parse arguments
lst_window_size=[int(x) for x in args.lst_window_size.split(',')]
lst_K=[int(x) for x in args.lst_K.split(',')]
lst_lags=[int(x) for x in args.lst_lags.split(',')]
l1, l2, l3, l4 = lst_lags
max_horizon = args.max_horizon
start = args.start
gamma = args.gamma

# Read tensor data and parameters
X = jnp.load(f'{dir_input}/mat_ptf_re_lag_{max_lag}.npz')['mat_ptf_re_rank'] # dim: (T, max_lag, num_ptf)
params = jnp.load(f'{dir_input}/dict_param_lag_{max_lag}.pkl', allow_pickle=True)
num_ptf = X.shape[-1]
assert params['max_lag'] == max_lag == args.max_lag

if config == 'char_anom':
    bin_labels, _, _, max_lag, frac_longshort, all_dates, start_date_maxlag = params.values()
elif config == 'wrds' or config == 'scs':
    bin_labels, all_dates = params['lst_char'], params['all_dates']


# FIX WINDOW SIZE, NOW TRYING WITH VARYING LAGS
window_size = 120
# start_date_oos parameter can be changed in the _load_configs.py file
dates = all_dates[-len(all_dates[all_dates >= start_date_oos]) - window_size:] if start != 'default' else all_dates
start_year = str(dates[window_size])[:4]


assert not jnp.isnan(X).any()
X_log = jnp.log(1 + X)[-len(dates):]

dict_pooled_pca_oos = {}
in_axes = (None, 0, None, None, None, None, None)
Pooled_PCA_Multiperiod = jax.vmap(Pooled_PCA_Multiperiod_Unmapped, in_axes=in_axes)
lag_to_chunk = {36: 120, 60: 60, 90: 25, 120: 10} if config != 'wrds' else {36: 10, 60: 5, 90: 2, 120: 2}
ub = X_log.shape[0] - window_size - max_horizon + 1

for lag in lst_lags:
    print(f"Processing lag = {lag}, Horizon = {max_horizon}")
    pooled_pca_lst = []
    pbar = tqdm(total=len(lst_K))
    for K in lst_K: # can i speed this up using jax.lax.scan?
        print(f"Window size = {window_size}, Lag = {lag}, K = {K}")
        if lag in lag_to_chunk:
            chunks = []
            for i in range(0, ub, lag_to_chunk[lag]):
                chunk = Pooled_PCA_Multiperiod(X_log[:, :lag, :], jnp.arange(i, min(lag_to_chunk[lag], ub)), K, window_size, lag, max_horizon, gamma) # dim: (num_windows, args.max_horizon)
                chunks.append(chunk)
            pooled_pca_lst.append(jnp.concatenate(chunks, axis=0)[..., None])
        else:
            x = Pooled_PCA_Multiperiod(X_log[:, :lag, :], jnp.arange(ub), K, window_size, lag, max_horizon, gamma) # dim: (num_windows, args.max_horizon)
            pooled_pca_lst.append(jnp.expand_dims(x, axis=-1))

        pbar.update(1)
    # Stores results
    dict_pooled_pca_oos[lag] = jnp.concatenate(pooled_pca_lst, axis=-1)


# Output directory
dir_out = f'../results_oos/multiperiod/{config}/pooled_pca_fig_oos_{input_type}_{spec}_ver{idx_ver}/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Save returns output
with open(dir_out + f'dict_pooled_pca_oos_{start_year}.pkl', 'wb') as handle:
    pickle.dump(dict_pooled_pca_oos, handle, protocol=pickle.HIGHEST_PROTOCOL)

for method in ["PooledPCA"]:
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{config.upper()} Pooled PCA Multiperiod Results', fontsize=14)
    for lag, (x, y) in zip(lst_lags, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        sr = dict_pooled_pca_oos[lag].mean(axis=0) / dict_pooled_pca_oos[lag].std(axis=0)
        start_date, end_date = str(dates[window_size])[:10], str(dates[-max_horizon - 1])[:10]
        plot2x2(sr, lst_K, max_horizon, method, window_size, lag, x, y, fig, axes, start_date, end_date)

    filename = f'multiperiod_ver{idx_ver}_Horizon{max_horizon}_{start_date}_{method}'
    fig.savefig(f'{dir_out}{filename}', bbox_inches='tight')