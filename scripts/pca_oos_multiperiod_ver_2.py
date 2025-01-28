"""
========================================================================================================================
Example usage: 
python3 pca_oos_multiperiod_ver_2.py --dataset=scs --lst_window_size=120 --lst_K=1,3,5,10,15,20,25 --max_horizon=36 --max_lag=120 --start='01-2005'
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
parser.add_argument('--max_lag', type=int, default=60, help='max lag' )
parser.add_argument('--lst_gamma', type=str, default='-1,0,10,20', help='gamma RPPCA values' )
# parser.add_argument('--lst_lags', type=str, default='12,24,36,60')
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
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 1 means Titan and FP64 mode, 
jax.config.update('jax_platform_name', 'gpu')
print("JAX is using device:", jax.devices()[0], jax.devices())

import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from typing import List, Tuple
from collections import defaultdict
from tfm.utils._load_configs import *
from tfm.models.pca import PCA_Multiperiod_Unmapped, ModelFree_Multiperiod_Unmapped
from tfm.utils._eval import *

# Further parse arguments
lst_window_size=[int(x) for x in args.lst_window_size.split(',')]
lst_K=[int(x) for x in args.lst_K.split(',')]
lst_gamma=[int(x) for x in args.lst_gamma.split(',')]
g1, g2, g3, g4 = lst_gamma
max_horizon = args.max_horizon
start = args.start

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

dict_pca_oos = defaultdict(dict) # maps {gamma: {K: matrix of shape (gammas, time)}}
in_axes = (None, 0, None, None, None, None)
PCA_Multiperiod_no_gamma = jax.vmap(PCA_Multiperiod_Unmapped, in_axes=in_axes)
PCA_Multiperiod = jax.vmap(PCA_Multiperiod_no_gamma, in_axes=(None, None, 0, None, None, None))

Modelfree_Multiperiod = jax.vmap(ModelFree_Multiperiod_Unmapped, in_axes=(None, 0, None, None))

for window_size in lst_window_size:
    print(f"Processing lag = {window_size}, Horizon = {max_horizon}")
    pbar = tqdm(total=max_horizon * len(lst_K))
    for horizon in range(1, 1 + max_horizon): # can i speed this up using jax.lax.scan?
        gamma1_lst, gamma2_lst, gamma3_lst, gamma4_lst = [], [], [], []
        ub = X_log.shape[0] - window_size - horizon + 1

        # Get model free results
        model_free = Modelfree_Multiperiod(X_log, jnp.arange(ub), window_size, horizon)
        dict_pca_oos['model_free'][horizon] = model_free

        # Get PCA results
        for K in lst_K:
            print(f"Window size = {window_size}, horizon = {horizon}, K = {K}")
            x1, x2, x3, x4 = PCA_Multiperiod(X_log, jnp.arange(ub), jnp.array(lst_gamma), K, window_size, horizon) # dim: (len(lst_gammas), num_windows)

            gamma1_lst.append(jnp.expand_dims(x1, axis=0))
            gamma2_lst.append(jnp.expand_dims(x2, axis=0))
            gamma3_lst.append(jnp.expand_dims(x3, axis=0))
            gamma4_lst.append(jnp.expand_dims(x4, axis=0))

            pbar.update(1)

        # Stores results
        dict_pca_oos[g1][horizon] = jnp.concatenate(gamma1_lst, axis=0) # dim: (len(lst_K), num_windows)
        dict_pca_oos[g2][horizon] = jnp.concatenate(gamma2_lst, axis=0)
        dict_pca_oos[g3][horizon] = jnp.concatenate(gamma3_lst, axis=0)
        dict_pca_oos[g4][horizon] = jnp.concatenate(gamma4_lst, axis=0)


# Output directory
dir_out = f'../results_oos/multiperiod/{config}/pca_fig_oos_{input_type}_{spec}_ver{idx_ver}/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# # Save returns output
with open(dir_out + f'dict_pca_oos_{start_year}.pkl', 'wb') as handle:
    pickle.dump(dict_pca_oos, handle, protocol=pickle.HIGHEST_PROTOCOL)

# dict_pca_oos = jnp.load(dir_out + f'dict_pca_oos_{start_year}.pkl', allow_pickle=True)

# Create a figure with two subplots side by side
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
start_date, end_date = str(dates[window_size])[:10], str(dates[-max_horizon - 1])[:10]
fig.suptitle(f'{config.upper()} RP-PCA Multiperiod Results, {start_date} to {end_date}', fontsize=14)
for gamma, (x, y) in zip(lst_gamma, [(0, 0), (0, 1), (1, 0), (1, 1)]):
    sr_lst = []
    model_free = []
    for horizon in range(1, max_horizon + 1):
        sr = dict_pca_oos[gamma][horizon].mean(axis=1) / dict_pca_oos[gamma][horizon].std(axis=1)
        sr_lst.append(jnp.expand_dims(sr, axis=0))

        model_free_sr = dict_pca_oos['model_free'][horizon].mean() / dict_pca_oos['model_free'][horizon].std() # dim: scalar
        model_free.append(jnp.expand_dims(model_free_sr, axis=0))

    sr = jnp.concatenate(sr_lst, axis=0) # dim: (max_horizon, len(lst_K))
    model_free = jnp.concatenate(model_free, axis=0) # dim: (max_horizon)
    plot_sr = jnp.concatenate([sr, model_free[..., None]], axis=1)
    pca_plot2x2(plot_sr, lst_K + ['Model Free'], max_horizon, f'RP-PCA, Gamma={gamma}', window_size, 0, x, y, fig, axes)

filename = f'multiperiod_ver{idx_ver}_Horizon{max_horizon}_{start_date}_{gamma}'
fig.savefig(f'{dir_out}{filename}', bbox_inches='tight')

# for method in ["PCA", "RPPCA"]:
#     # Create a figure with two subplots side by side
#     fig, axes = plt.subplots(2, 2, figsize=(14, 12))
#     fig.suptitle(f'{config.upper()} Multiperiod Results', fontsize=14)
#     for lag, (x, y) in zip(lst_lags, [(0, 0), (0, 1), (1, 0), (1, 1)]):
#         sr = dict_pca_oos[method][lag].mean(axis=0) / dict_pca_oos[method][lag].std(axis=0)
#         start_date, end_date = str(dates[window_size])[:10], str(dates[-max_horizon - 1])[:10]
#         plot2x2(sr, lst_K, max_horizon, method, window_size, lag, x, y, fig, axes, start_date, end_date)

#     filename = f'multiperiod_ver{idx_ver}_Horizon{max_horizon}_{start_date}_{method}'
#     fig.savefig(f'{dir_out}{filename}', bbox_inches='tight')