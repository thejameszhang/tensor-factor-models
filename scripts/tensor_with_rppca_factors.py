"""
========================================================================================================================
Example usage: 
python3 tensor_with_rppca_factors.py --dataset=wrds --lst_window_size=120 --lst_K=1,2,3,4,5,10,15,20,25 --lst_lags=36,60,90,120 --max_horizon=36 --max_lag=120
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
parser.add_argument('--lasso', type=float, default=0)
args = parser.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

#%%
# from types import SimpleNamespace
# args = SimpleNamespace(dataset='char_anom', lst_K='1,3,5,10,20,30', 
#                        lst_window_size='60,120', max_horizon=36)

config = args.dataset
assert args.dataset in ('char_anom', 'wrds', 'scs', 'ff', 'toy')
os.environ['CONFIG'] = config

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
os.environ['CUDA_VISIBLE_DEVICES'] = "1" # Titan and FP64 mode, 2 is V100?
jax.config.update('jax_platform_name', 'gpu')
print("JAX is using device:", jax.devices()[0], jax.devices())

import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pickle
from tfm.utils._load_configs import *
from tfm.models.tensor import Tensor_Model_With_RPPCA_Factors
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

in_axes = (None, 0, None, None, None, None) 
Tensor_Models = jax.vmap(Tensor_Model_With_RPPCA_Factors, in_axes=in_axes)

# lag_to_chunk = {36: 80, 60: 10, 90: 5, 120: 3} # wrds setting
lag_to_chunk = {36: 5, 60: 2, 90: 2, 120: 2} # wrds setting
ub = X_log.shape[0] - window_size - max_horizon + 1

dict_tensor_oos = {
    "TFM with RPPCA Factors": {}, # for each lag, dim: (T, max_horizon, len(lst_K))
    "W": {lag: defaultdict(dict) for lag in lst_lags}, # for each (lag, K) pair, dim: (T, lag, K)
    "F": {lag: defaultdict(dict) for lag in lst_lags}, # for each (lag, K) pair, dim: (T, window_size, K)
    "B": {lag: defaultdict(dict) for lag in lst_lags}, # for each (lag, K) pair, dim: (T, window_size, K)
}

pbar = tqdm(total=len(lst_K) * len(lst_lags))
for lag in lst_lags:
    print(f"Processing lag = {lag}, Horizon = {max_horizon}")
    tfm_lst = []
    for K in lst_K: # can i speed this up using jax.lax.scan?
        print(f"Window size = {window_size}, Lag = {lag}, K = {K}")
        if lag in lag_to_chunk and config == 'wrds':
            model_chunks, W_chunks, F_chunks, B_chunks = [], [], [], []
            for i in range(0, ub, lag_to_chunk[lag]):
                model, W, F, B = Tensor_Models(X_log[:, :lag, :], jnp.arange(i, min(i + lag_to_chunk[lag], ub)), K, window_size, lag, max_horizon) # dim: (num_windows, args.max_horizon)
                model_chunks.append(model)
                W_chunks.append(W)
                F_chunks.append(F)
                B_chunks.append(B)

            tfm_lst.append(jnp.concatenate(model_chunks, axis=0)[..., None])
            dict_tensor_oos['W'][lag][K] = jnp.concatenate(W_chunks, axis=0)
            dict_tensor_oos['F'][lag][K] = jnp.concatenate(F_chunks, axis=0)
            dict_tensor_oos['B'][lag][K] = jnp.concatenate(B_chunks, axis=0)
        # else:
        # model, W, F, B = Tensor_Models(X_log[:, :lag, :], jnp.arange(ub), K, window_size, lag, max_horizon) # dim: (num_windows, args.max_horizon)
        # tfm_lst.append(jnp.expand_dims(model, axis=-1))
        # dict_tensor_oos['W'][lag][K] = W
        # dict_tensor_oos['F'][lag][K] = F
        # dict_tensor_oos['B'][lag][K] = B
        pbar.update(1)

    # Stores results
    dict_tensor_oos["TFM with RPPCA Factors"][lag] = jnp.concatenate(tfm_lst, axis=-1)


# Output directory
dir_out = f'../results_oos/multiperiod/{config}/HYBRID_tensor_fig_oos_ret_rankptf_ver{idx_ver}/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Save returns output
with open(dir_out + f'saved_dict_tensor_oos_{start_year}.pkl', 'wb') as handle:
    pickle.dump(dict_tensor_oos, handle, protocol=pickle.HIGHEST_PROTOCOL)


for method in ["TFM with RPPCA Factors"]:
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{config.upper()} Multiperiod Results', fontsize=14)
    for lag, (x, y) in zip(lst_lags, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        sr = dict_tensor_oos[method][lag].mean(axis=0) / dict_tensor_oos[method][lag].std(axis=0)
        start_date, end_date = str(dates[window_size])[:10], str(dates[-max_horizon - 1])[:10]
        plot2x2(sr, lst_K, max_horizon, method, window_size, lag, x, y, fig, axes, start_date, end_date)

    filename = f'{method}_ver{idx_ver}_Horizon36_{start_year}'
    fig.savefig(f'{dir_out}{filename}.png', bbox_inches='tight')