"""
========================================================================================================================
Example usage: 
python3 pooled_pca_oos_multiperiod.py --dataset=scs --lst_window_size=120 --lst_K=1,2,3,4,5,10,15,20,25 --lst_lags=36,60,90,120 --max_horizon=36 --max_lag=120
========================================================================================================================

Author: James Zhang
Date: November 2024
"""
import argparse
import os
import pprint

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='scs', help='char_anom, scs, wrds, ff' )
parser.add_argument('--lst_K', type=str, default='1,3,5,10,20',
    help='number of factors separated by ","' )
parser.add_argument('--lst_window_size', type=str, default='60,120,240',
    help='list of rolling window sizes separated by ","')
parser.add_argument('--max_horizon', type=int, default=36, 
    help='max horizon in multiperiod portfolio' )
parser.add_argument('--max_lag', type=int, default=60, help='max lag' )
parser.add_argument('--lst_lags', type=str, default='12,24,36,60')
parser.add_argument('--start', type=str, default='default')
args = parser.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

config = args.dataset
assert args.dataset in ('char_anom', 'wrds', 'scs', 'ff', 'toy')
os.environ['CONFIG'] = config

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
os.environ['CUDA_VISIBLE_DEVICES'] = "2" # Titan and FP64 mode, 2 is V100?
jax.config.update('jax_platform_name', 'gpu')
print("JAX is using device:", jax.devices()[0], jax.devices())

import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pickle
from tfm.utils._load_configs import *
from tfm.models.pca import Pooled_PCA_Multiperiod_Unmapped
from tfm.utils._eval import *

# Further parse arguments
lst_window_size=[int(x) for x in args.lst_window_size.split(',')]
lst_K=[int(x) for x in args.lst_K.split(',')]
lst_lags=[int(x) for x in args.lst_lags.split(',')]
max_horizon = args.max_horizon
start = args.start

# Read tensor data and parameters
X = jnp.load(f'{dir_input}/mat_ptf_re_lag_{max_lag}.npz')['mat_ptf_re_rank'] # dim: (T, max_lag, num_ptf)
params = jnp.load(f'{dir_input}/dict_param_lag_{max_lag}.pkl', allow_pickle=True)
assert params['max_lag'] == max_lag == args.max_lag

if config == 'char_anom':
    bin_labels, _, _, max_lag, frac_longshort, all_dates, start_date_maxlag = params.values()
else:
    bin_labels, all_dates = params['lst_char'], params['all_dates']

# FIX WINDOW SIZE, NOW TRYING WITH VARYING LAGS
window_size = 120
all_dates = all_dates[max_lag - 1:] if config != 'char_anom' else all_dates
dates = all_dates[-len(all_dates[all_dates >= start_date_oos]) - window_size:] if start != 'default' else all_dates
start_year = str(dates[window_size])[:4]
assert not jnp.isnan(X).any()

print(X.shape) # should be like 475

X_log = jnp.log(1 + X)[-len(dates):]

in_axes = (None, 0, None, None, None, None, None)
PooledPCA = jax.vmap(Pooled_PCA_Multiperiod_Unmapped, in_axes=in_axes)
# lag_to_chunk = {60: 150, 90: 80, 120: 25} # scs setting
lag_to_chunk = {36: 10, 60: 3, 90: 3, 120: 2} # wrds setting
ub = X_log.shape[0] - window_size - max_horizon + 1

dict_ppca_oos = {
    "Returns": {}, # for each lag, dim: (T, max_horizon, len(lst_K))
    "F": {lag: defaultdict(dict) for lag in lst_lags}, # for each (lag, K) pair, dim: (T, window_size, K)
    "Lambda": {lag: defaultdict(dict) for lag in lst_lags}, # for each (lag, K) pair, dim: (T, window_size, K)
    "Weights": {lag: defaultdict(dict) for lag in lst_lags} # for each (lag, K) pair, dim: (T, max_horizon, K)
}

pbar = tqdm(total=len(lst_K) * len(lst_lags))
for lag in lst_lags:
    print(f"Processing lag = {lag}, Horizon = {max_horizon}")
    ppca_lst = []
    for K in lst_K: # can i speed this up using jax.lax.scan?
        print(f"Window size = {window_size}, Lag = {lag}, K = {K}")
        if lag in lag_to_chunk:
            model_chunks, F_chunks, Lambda_chunks, weight_chunks = [], [], [], []
            for i in range(0, ub, lag_to_chunk[lag]):
                model, F, Lambda, weights = PooledPCA(X_log[:, :lag, :], jnp.arange(i, min(i + lag_to_chunk[lag], ub)), K, window_size, lag, max_horizon, -1) # dim: (num_windows, args.max_horizon)
                model_chunks.append(model)
                F_chunks.append(F)
                Lambda_chunks.append(Lambda)
                weight_chunks.append(weights)

            ppca_lst.append(jnp.concatenate(model_chunks, axis=0)[..., None])
            dict_ppca_oos['F'][lag][K] = jnp.concatenate(F_chunks, axis=0)
            dict_ppca_oos['Lambda'][lag][K] = jnp.concatenate(Lambda_chunks, axis=0)
            dict_ppca_oos['Weights'][lag][K] = jnp.concatenate(weight_chunks, axis=0)
        else:
            model, F, Lambda, weights = PooledPCA(X_log[:, :lag, :], jnp.arange(ub), K, window_size, lag, max_horizon, -1) # dim: (num_windows, args.max_horizon)
            ppca_lst.append(jnp.expand_dims(model, axis=-1))
            dict_ppca_oos['F'][lag][K] = F
            dict_ppca_oos['Lambda'][lag][K] = Lambda
            dict_ppca_oos['Weights'][lag][K] = weights
        pbar.update(1)

    # Stores results
    dict_ppca_oos["Returns"][lag] = jnp.concatenate(ppca_lst, axis=-1)


# Output directory
dir_out = f'../results_oos/multiperiod/{config}/pooled_pca_fig_oos_ret_rankptf_ver{idx_ver}/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Save returns output
with open(dir_out + f'saved_dict_ppca_oos_{start_year}.pkl', 'wb') as handle:
    pickle.dump(dict_ppca_oos, handle, protocol=pickle.HIGHEST_PROTOCOL)

method = "Returns"
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'{config.upper()} Multiperiod Results', fontsize=14)
for lag, (x, y) in zip(lst_lags, [(0, 0), (0, 1), (1, 0), (1, 1)]):
    sr = dict_ppca_oos[method][lag].mean(axis=0) / dict_ppca_oos[method][lag].std(axis=0)
    start_date, end_date = str(dates[window_size])[:10], str(dates[-max_horizon - 1])[:10]
    plot2x2(sr, lst_K, max_horizon, "PooledPCA", window_size, lag, x, y, fig, axes, start_date, end_date)

filename = f'{method}_ver{idx_ver}_Horizon36_{start_year}'
fig.savefig(f'{dir_out}{filename}.png', bbox_inches='tight')