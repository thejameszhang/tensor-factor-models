import jax
import jax.numpy as jnp
from typing import Tuple, List, Optional, Dict
from functools import partial
from tfm.utils._constants import *

@partial(jax.jit, backend=main_compute_device)
def compute_Z_row(W: WTensor, B: BTensor, S: STensor, i: int):
    """Compute Z_fit for all K components at once"""
    return jnp.kron(W[:, i], B[:, i]) * S[i]

# @partial(jax.jit, backend=main_compute_device)
# def get_F_next(Z_fit: jnp.ndarray, X_log: jnp.ndarray, idx_h: int, idx_window: int, window_size: int):
#     """Get F_next for the next max_horizon periods by regressing X_next on tensor loadings"""
#     X_next = X_log[idx_window + window_size + idx_h]
#     X_next_flatten = X_next.reshape(1, -1)
#     mat_weight_flatten = Z_fit.T @ jnp.linalg.inv(Z_fit @ Z_fit.T) # dim: (NL, K)
#     F_next = X_next_flatten @ mat_weight_flatten
#     return jnp.squeeze(F_next, axis=0)

@partial(jax.jit, backend=main_compute_device)
def get_F_next(Z_fit: jnp.ndarray, X_log: jnp.ndarray, idx_h: int):
    """Get F_next for the next max_horizon periods by regressing X_next on tensor loadings"""
    X_next = X_log[idx_h]
    X_next_flatten = X_next.reshape(1, -1) # dim: (1, NL)
    mat_weight_flatten = Z_fit.T @ jnp.linalg.inv(Z_fit @ Z_fit.T) # dim: (NL, K)
    F_next = X_next_flatten @ mat_weight_flatten
    return jnp.squeeze(F_next, axis=0) # dim: (K,)

@partial(jax.jit, backend=main_compute_device)
def get_F_oos(Z_fit: jnp.ndarray, X_oos: jnp.ndarray, idx_t: int):
    """Get F_next for all oos times at once"""
    X_oos_flatten = X_oos[idx_t].reshape(1, -1)
    mat_weight_flatten = Z_fit.T @ jnp.linalg.inv(Z_fit @ Z_fit.T)
    return X_oos_flatten @ mat_weight_flatten

# @partial(jax.jit, backend=main_compute_device)
# def get_multiperiod_return(F: FTensor, W: WTensor, idx_k: int):
#     """Get approximate multiperiod return in the window"""
#     return jnp.cumsum(F[:, idx_k][:, jnp.newaxis] @ W[:, idx_k][jnp.newaxis, :], axis=1)

@partial(jax.jit, backend=main_compute_device)
def outerW(W: WTensor, i: int):
    """Approximate covariance matrix at all lookback times up to max lag."""
    w = W[i, :][..., None]
    return w @ w.T
    
@partial(jax.jit, backend=main_compute_device)
def get_cov_approx(FW: jnp.ndarray, idx_s: int):
    """Approximate covariance matrix at all lookback times up to max lag."""
    return jnp.cov(FW[:, :, idx_s], bias=True)

@partial(jax.jit, backend=main_compute_device, static_argnums=(2))
def get_mv_weights(mu: jnp.ndarray, cov: jnp.ndarray, K: int, idx_s: int):
    if K > 1:
        return jnp.linalg.inv(cov[idx_s]) @ mu[idx_s]
    else:
        return jnp.mean(mu) / jnp.var(cov)
    
@partial(jax.jit, backend=main_compute_device)
def get_X_fitted_oos(W: WTensor, B: BTensor, S: STensor, F_oos: jnp.ndarray, i: int):
    """"""
    BWS = (W[:, i][:, jnp.newaxis] @ B[:, i][:, jnp.newaxis].T) * S[i]
    return F_oos[:, i][:, jnp.newaxis, jnp.newaxis] * BWS


@partial(jax.jit, backend=main_compute_device, static_argnums=(2, 3, 4,))
def get_multiperiod_return(F: FTensor, W: WTensor, max_horizon: int, K: int, window_size: int):
    """
    Computes multi-horizon returns for K basis assets by summing factors and loadings over
    for one month. Note this is not from month t, we are computing returns for month
    t + S. 
    """
    
    @partial(jax.jit, backend=main_compute_device, static_argnums=(1,))
    def _process(t: int, max_horizon: int):
        """
        At month t + S, aligns factor returns with the correct lag and returns 
        multi-horizon returns. 
        """
        F_temp = jax.lax.dynamic_slice(F, start_indices=(t, 0), slice_sizes=(max_horizon, K))
        # Flip reverses the time dimension, is this needed, i think so
        # return jnp.sum(jnp.flip(F_temp, axis=0) * W[:max_horizon], axis=0)
        # return jnp.cumsum(jnp.flip(F_temp, axis=0) * W[:max_horizon], axis=0) # dim: (max_horizon, K)
        return jnp.cumsum(F_temp * W[:max_horizon], axis=0) # dim: (max_horizon, K)

    # This doesn't leverage all possible data in the rolling window for some horizons, but will this affect results?
    FW = jax.vmap(_process, in_axes=(0, None))(jnp.arange(window_size - max_horizon + 1), max_horizon)
    return FW

@partial(jax.jit, backend=main_compute_device)
def get_mu_naive(FW: jnp.ndarray, idx_h: int):
    return jnp.mean(FW[:, idx_h, :], axis=0)
    

