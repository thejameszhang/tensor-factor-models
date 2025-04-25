import jax
import jax.numpy as jnp
import time
from tfm.parafac_orthogonal import orth_als, parafac_orthogonal, reconstruct_tensor
from tfm.parafac_jax import parafac_enhanced
import pandas as pd

start_date_oos = pd.to_datetime('2005-01-01')
window_size = 120
max_lag = 120
start = 'default'
max_horizon = 36
lst_K = [1, 2, 3, 4, 5, 10, 15, 20, 25]
lst_lags = [36, 60, 90, 120]

dataset = 'wrds'
params = jnp.load(f'../organized_data/organized_data/{dataset}/dict_param_lag_120.pkl', allow_pickle=True)
bin_labels, all_dates = params['lst_char'], params['all_dates']
all_dates = all_dates[max_lag - 1:]
dates = all_dates[-len(all_dates[all_dates >= start_date_oos]) - window_size:] if start != 'default' else all_dates
trade_days = dates[window_size - 1: -max_horizon]

X = jnp.load(f'../organized_data/organized_data/{dataset}/mat_ptf_re_lag_120.npz')['mat_ptf_re_rank']
X_log = jnp.log(1 + X)

tensor = X_log
rank = 20
n_iter_max = 100
random_state = 100

# Measure runtime and reconstruction error for orth_als
start_time = time.time()
weights_orth_als, factors_orth_als = orth_als(tensor, rank, n_iter_max, random_state)
end_time = time.time()
# reconstruction_orth_als = jnp.sum(jnp.array([jnp.outer(factors_orth_als[0][:, i], jnp.outer(factors_orth_als[1][:, i], factors_orth_als[2][:, i])) for i in range(rank)]), axis=0)
reconstruction_orth_als = reconstruct_tensor(weights_orth_als, factors_orth_als)
error_orth_als = jnp.linalg.norm(tensor - reconstruction_orth_als) / jnp.linalg.norm(tensor)
time_orth_als = end_time - start_time

# Measure runtime and reconstruction error for parafac_orthogonal
start_time = time.time()
weights_ortho, factors_ortho = parafac_orthogonal(tensor, rank, random_state=random_state, n_iter_max=n_iter_max, fixed_intercept_mode=-1)
end_time = time.time()
# reconstruction_parafac = jnp.sum(jnp.array([jnp.outer(factors_parafac[0][:, i], jnp.outer(factors_parafac[1][:, i], factors_parafac[2][:, i])) for i in range(rank)]), axis=0)
reconstruction_parafac = reconstruct_tensor(weights_ortho, factors_ortho)
error_parafac = jnp.linalg.norm(tensor - reconstruction_parafac) / jnp.linalg.norm(tensor)
time_parafac = end_time - start_time


# Measure runtime and reconstruction error for parafac_orthogonal
start_time = time.time()
weights, factors = parafac_enhanced(tensor, rank, random_state=random_state, n_iter_max=n_iter_max, fix_intercept_mode=-1)
end_time = time.time()
# reconstruction_parafac = jnp.sum(jnp.array([jnp.outer(factors_parafac[0][:, i], jnp.outer(factors_parafac[1][:, i], factors_parafac[2][:, i])) for i in range(rank)]), axis=0)
reconstruction = reconstruct_tensor(weights, factors)
error = jnp.linalg.norm(tensor - reconstruction) / jnp.linalg.norm(tensor)
time = end_time - start_time

# Print results
print(f"Orth-ALS Runtime: {time_orth_als:.4f} seconds")
print(f"Orth-ALS Reconstruction Error: {error_orth_als:.4e}")
print(f"PARAFAC-Orthogonal Runtime: {time_parafac:.4f} seconds")
print(f"PARAFAC-Orthogonal Reconstruction Error: {error_parafac:.4e}") 
print(f"PARAFAC Runtime: {time:.4f} seconds")
print(f"PARAFAC Reconstruction Error: {error:.4e}") 