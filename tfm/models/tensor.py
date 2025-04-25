from tqdm import tqdm
import numpy as np
from functools import partial
from typing import List, Optional, Tuple, Dict
from functools import partial
import jax
import jax.numpy as jnp
from tfm.parafac_jax import parafac_enhanced, normalize_factors
# from tfm.parafac_admm import parafac_admm
from tfm.parafac_orthogonal import *
from tfm.utils._tensor import *
from tfm.utils._constants import *
from tfm.utils._cov import newey_west_cov_jax
from tfm.utils._pca import RPPCA, get_multiperiod_returns

@partial(jax.jit, backend=main_compute_device, static_argnums=(2, 3, 4, 5, 6))
def Tensor_Multiperiod_Unmapped_Monthly(X_log: jnp.ndarray, 
                                idx_window: int, 
                                K: int, 
                                window_size: int,
                                lag: int,
                                max_horizon: int,
                                lasso: Tuple[float]):
    """
    Computes mean variance weights and portfolio returns in a multiperiod 
    setting for one K and one window size. K and window_size MUST be static. Done in a rolling 
    window fashion. idx_window represents the point of oldest data in the rolling window. The function
    essentially acts as an investor from the perspective at month idx_window + window_size. Note that 
    window_size and lag must be greater than K, the rank oif the PARAFAC decomposition.
    Args:
        X_log: log returns - dim: (T, max_lag, N)
        idx_window: 
        K: rank of PARAFAC decomposition
        window_size: 120
    Returns:
        ret_MV: mean variance factor returns - dim: (max_horizon,)
    """
    T, _, N  = X_log.shape
    X_fit = jax.lax.dynamic_slice(X_log, start_indices=(idx_window, 0, 0), slice_sizes=(window_size, lag, N))
    X_next = jax.lax.dynamic_slice(X_log, start_indices=(idx_window + window_size, 0, 0), slice_sizes=(max_horizon, lag, N))
    
    weights, factors = parafac_enhanced(
        tensor=X_fit,
        rank=K,
        random_state=random_seed,
        n_iter_max=n_iter_max,
        fix_intercept_mode=1,
    )
    
    # Extract and normalize factors
    factors = dict(zip(['F','W','B'], factors))
    factors['S'] = weights
    factors = normalize_factors(factors, reorder=True)
    F, W, B, S = [factors[key] for key in ['F','W','B','S']] 

    Z_fit = jax.vmap(compute_Z_row, in_axes=(None, None, None, 0))(W, B, S, jnp.arange(K)) # dim: (K, NL)
    F_next = jax.vmap(get_F_next, in_axes=(None, None, 0))(Z_fit, X_next, jnp.arange(max_horizon)) # dim: (max_horizon, K)
    FW_next = (F_next * W[:max_horizon, :]).cumsum(axis=0) # dim: (max_horizon, K)
    
    # 1. Get mean variance weights on basis assets - approximate solution using the tensor model 
    mu_tfm = jnp.multiply(jnp.cumsum(W, axis=0), jnp.mean(F, axis=0)) # dim: (lag, K)
    var_tfm = jnp.cumsum(jax.vmap(outerW, in_axes=(None, 0))(W, jnp.arange(lag)), axis=0) * jnp.cov(F.T, bias=True) # dim: (lag, K, K)
    mv_tfm = jax.vmap(get_mv_weights, in_axes=(None, None, None, 0))(mu_tfm, var_tfm, K, jnp.arange(max_horizon)) # dim: (max_horizon, K)
    ret_tfm = (FW_next * mv_tfm).sum(axis=1) # dim: (max_horizon)
    
    # 2. Naive method
    FW = get_multiperiod_return(F, W, max_horizon, K, window_size) # dim: (window_size - max_horizon + 1, max_horizon, K)
    mu_naive = jax.vmap(get_mu_naive, in_axes=(None, 0))(FW, jnp.arange(max_horizon)) # dim: (max_horizon, K)
    var_naive = jnp.concatenate([jnp.expand_dims(newey_west_cov_jax(FW[:, i, :], num_overlap=i), axis=0) for i in range(max_horizon)], axis=0)
    mv_naive = jax.vmap(get_mv_weights, in_axes=(None, None, None, 0))(mu_naive, var_naive, K, jnp.arange(max_horizon)) # dim: (max_horizon, K)
    ret_naive = (FW_next * mv_naive).sum(axis=1) # dim: (max_horizon)

    # 3. Use MVE weights for the 1-month horizon, but hold this for all horizons
    mv_tfm_naive = jnp.tile(mv_tfm[0][None, :], (max_horizon, 1)) # dim: (max_horizon, K)
    ret_tfm_naive = (FW_next * mv_tfm_naive).sum(axis=1) # dim: (max_horizon)

    # 5. Markowitz
    if K > 1:
        mu_markowitz = jnp.mean(F, axis=0) # dim: (K,)
        var_markowitz = jnp.cov(F.T) # dim: (K, K)
        mv_markowitz = jnp.linalg.inv(var_markowitz) @ mu_markowitz # dim: (K,)
    else:
        mv_markowitz = jnp.expand_dims(jnp.mean(F) / jnp.var(F), axis=-1)
    ret_markowitz = (FW_next * mv_markowitz).sum(axis=1) # dim: (max_horizon)

    # 4. If factors are orthogonal, covariances are 0 and we're left with just variances 
    weights, factors = parafac_orthogonal(
        tensor=X_fit, 
        rank=K, 
        orthogonal_mode=0,
        fixed_intercept_mode=1,
        random_state=random_seed,
        n_iter_max=n_iter_max
    )
    # Extract and normalize factors
    factors = dict(zip(['F','W','B'], factors))
    factors['S'] = weights
    factors = normalize_factors(factors, reorder=True)
    Fo, Wo, Bo, So = [factors[key] for key in ['F','W','B','S']] 

    Z_fit = jax.vmap(compute_Z_row, in_axes=(None, None, None, 0))(Wo, Bo, So, jnp.arange(K)) # dim: (K, NL)
    Fo_next = jax.vmap(get_F_next, in_axes=(None, None, 0))(Z_fit, X_next, jnp.arange(max_horizon)) # dim: (max_horizon, K)
    FWo_next = (Fo_next * Wo[:max_horizon, :]).cumsum(axis=0) # dim: (max_horizon, K)
    mu_ortho = jnp.multiply(jnp.cumsum(Wo, axis=0), jnp.mean(Fo, axis=0)) # dim: (lag, K)
    mv_ortho = (mu_ortho / (jnp.var(Fo, axis=0) * jnp.cumsum(Wo * Wo, axis=0)))[:max_horizon] # dim: (max_horizon, K)
    ret_ortho = (FWo_next * mv_ortho).sum(axis=1) # dim: (max_horizon)

    # 5. Markowitz
    if K > 1:
        mv_markowitz_ortho = jnp.linalg.inv(jnp.cov(Fo.T)) @ jnp.mean(Fo, axis=0) # dim: (K,)
    else:
        mv_markowitz_ortho = jnp.expand_dims(jnp.mean(Fo) / jnp.var(Fo), axis=-1)
    ret_markowitz_ortho = (FWo_next * mv_markowitz_ortho).sum(axis=1) # dim: (max_horizon)
    
    return ret_tfm, ret_naive, ret_tfm_naive, ret_ortho, ret_markowitz, ret_markowitz_ortho, W, F, B, mv_tfm, mv_ortho, Wo, Fo, Bo

@partial(jax.jit, backend=main_compute_device, static_argnums=(2, 3, 4, 5))
def Tensor_Model_With_RPPCA_Factors(X_log: jnp.ndarray, 
                                idx_window: int, 
                                K: int, 
                                window_size: int,
                                lag: int,
                                max_horizon: int):

    T, _, N  = X_log.shape
    X_fit = jax.lax.dynamic_slice(X_log, start_indices=(idx_window, 0, 0), slice_sizes=(window_size, lag, N))
    X_next = jax.lax.dynamic_slice(X_log, start_indices=(idx_window + window_size, 0, 0), slice_sizes=(max_horizon, lag, N))
    
    factors_pca = RPPCA(X_fit.reshape(window_size, -1), gamma=-1, K=K)[0]
    
    weights, factors = parafac_enhanced(
        tensor=X_fit,
        rank=K,
        fixed_modes=(0, factors_pca),
        random_state=random_seed,
        n_iter_max=n_iter_max,
        fix_intercept_mode=1,
    )
    
    # Extract and normalize factors
    factors = dict(zip(['F','W','B'], factors))
    factors['S'] = weights
    factors = normalize_factors(factors, reorder=True)
    F, W, B, S = [factors[key] for key in ['F','W','B','S']] 

    Z_fit = jax.vmap(compute_Z_row, in_axes=(None, None, None, 0))(W, B, S, jnp.arange(K)) # dim: (K, NL)
    F_next = jax.vmap(get_F_next, in_axes=(None, None, 0))(Z_fit, X_next, jnp.arange(max_horizon)) # dim: (max_horizon, K)
    FW_next = (F_next * W[:max_horizon, :]).cumsum(axis=0) # dim: (max_horizon, K)
    
    # 1. Get mean variance weights on basis assets - approximate solution using the tensor model 
    mu_tfm = jnp.multiply(jnp.cumsum(W, axis=0), jnp.mean(F, axis=0)) # dim: (lag, K)
    var_tfm = jnp.cumsum(jax.vmap(outerW, in_axes=(None, 0))(W, jnp.arange(lag)), axis=0) * jnp.cov(F.T, bias=True) # dim: (lag, K, K)
    mv_tfm = jax.vmap(get_mv_weights, in_axes=(None, None, None, 0))(mu_tfm, var_tfm, K, jnp.arange(max_horizon)) # dim: (max_horizon, K)
    ret_tfm = (FW_next * mv_tfm).sum(axis=1) # dim: (max_horizon)
    
    return ret_tfm, W, F, B

def Tensor_One_Window_One_K(X_fit: jnp.ndarray, 
                            X_oos: jnp.ndarray, 
                            K: int, 
                            random_seed: int = 100):
    """
    Code needed to get the tensor term structure, alpha, unexplained variance graphs and tables. 
    Not done in a rolling window fashion.
    """
    T_fit, max_lag, num_ptf = X_fit.shape
    T_oos = X_oos.shape[0]
    weights, factors = parafac_enhanced(
        tensor=X_fit,
        rank=K,
        random_state=random_seed,
        n_iter_max=n_iter_max,
        fix_intercept_mode=1
    )
    
    # Extract and normalize factors
    factors = dict(zip(['F','W','B'], factors))
    factors['S'] = weights
    factors = normalize_factors(factors, reorder=True)
    F, W, B, S = [factors[key] for key in ['F','W','B','S']]

    # get F_next by regress X_next
    Z_fit = jax.vmap(compute_Z_row, in_axes=(None, None, None, 0))(W, B, S, jnp.arange(K))
    F_oos = jax.vmap(get_F_oos, in_axes=(None, None, 0))(Z_fit, X_oos, jnp.arange(T_oos))
    F_oos = jnp.squeeze(F_oos, axis=1) # dim: (T_oos, K)

    # mean-var portfolio construction. get scaler return (rx)
    # mat_ret_mv_oos, dim: (T_oos)
    if K>1:
        ret_mv_oos = F_oos @ np.linalg.inv(jnp.cov(F.T)) @ F.mean(axis=0)
    else:
        ret_mv_oos = F_oos*np.mean(F) / np.var(F)
    sr_oos = ret_mv_oos.mean(axis=0) / ret_mv_oos.std(axis=0)

    # get fit of next period return. expected_re is of dim (T_oos, num_ptf, max_lag)
    X_fitted_oos = jax.vmap(get_X_fitted_oos, in_axes=(None, None, None, None, 0))(W, B, S, F_oos, jnp.arange(K))
    X_fitted_oos = X_fitted_oos.sum(axis=0)

    return {
        'ret_mv_oos': ret_mv_oos, # dim: (T_oos,)
        'X_fitted_oos': X_fitted_oos, # dim: (T_oos, max_lag, num_ptf)  
        'sr_oos': sr_oos
    }
    