"""
PCA Models for Multiperiod Experiments
================================ 

================================
Author: James Zhang
Date: November 2024
"""

from tqdm import tqdm
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List
from tfm.utils._constants import *
from tfm.utils._pca import *

@partial(jax.jit, backend=main_compute_device, static_argnums=(3, 4, 5,))
def PCA_Multiperiod_Unmapped(X_log: jnp.ndarray, idx_window: int, gamma: int,
                            K: int, window_size: int, horizon: int = 36):
    """
    For each horizon h, constructs overlapping multiperiod returns and uses PCA to extract the K 
    factors. Computes mean variancce efficient weights and applies it to the OOS one-period return. 
    Args:
        - X_log: characteristics-based portfolio tensor - dim: (window_size, horizon, N)
    Returns:
        - returns: OOS one-period return - dim: (scalar,)
    """
    N = X_log.shape[2]
    X_fit = jax.lax.dynamic_slice(X_log, start_indices=(idx_window, 0, 0), slice_sizes=(window_size, horizon, N)) # dim: window_size, N
    X_multi_fit = get_multiperiod_returns(X_fit, horizon, window_size) # dim: (window_size - horizon + 1, N), no wasted observations
    
    factors_pca, loadings_pca = RPPCA(X_multi_fit, gamma=gamma, K=K) # dim: (T, K) and (N, K)
    mu_pca = jnp.mean(factors_pca, axis=0) # dim: (K,)
    var_pca = newey_west_cov_jax(factors_pca, num_overlap=horizon-1) # dim: (K, K)
    mv_pca = jnp.linalg.inv(var_pca) @ mu_pca # dim: (K,)
    
    X_next = jax.lax.dynamic_slice(X_log, start_indices=(idx_window + window_size, 0, 0), slice_sizes=(horizon, horizon, N)) # dim: 1, N
    X_multi_next = get_multiperiod_returns(X_next, horizon, horizon)[0] # dim: (N,)

    factors_next_pca = X_multi_next @ loadings_pca @ jnp.linalg.inv(loadings_pca.T @ loadings_pca) # dim: (1, K)
    ret_pca = factors_next_pca @ mv_pca # dim: scalar
    return ret_pca


@partial(jax.jit, backend=main_compute_device, static_argnums=(2, 3,))
def ModelFree_Multiperiod_Unmapped(X_log: jnp.ndarray, 
                            idx_window: int, 
                            window_size: int,
                            horizon: int = 36):
    """
    For each horizon h, constructs the same overlapping multihorizon return as above, but instead 
    of using PCA, simply computes the mean variance weights of the num_ptf characteristic-based 
    portfolios and applies this to the OOS one-period return. 
    Args:
        - X_log: T, max_lag, N
    Returns:
    """
    N = X_log.shape[2]
    X_fit = jax.lax.dynamic_slice(X_log, start_indices=(idx_window, 0, 0), slice_sizes=(window_size, horizon, N)) # dim: window_size, N
    X_multi_fit = get_multiperiod_returns(X_fit, horizon, window_size) # dim: window_size - horizon + 1, N
    mu = jnp.mean(X_multi_fit, axis=0) # dim: (N,)
    var = newey_west_cov_jax(X_multi_fit, num_overlap=horizon-1) # dim: (N, N)
    mv = jnp.linalg.inv(var) @ mu # dim: (N,)
    X_next = jax.lax.dynamic_slice(X_log, start_indices=(idx_window + window_size, 0, 0), slice_sizes=(horizon, horizon, N)) # dim: 1, N
    X_multi_next = get_multiperiod_returns(X_next, horizon, horizon)[0] # dim: (N,)
    ret = X_multi_next @ mv  # dim: scalar, could just do 
    return ret # dim: scalar


@partial(jax.jit, backend=main_compute_device, static_argnums=(3, 4, 5,))
def Toy_PCA_Unmapped(X_log: jnp.ndarray, idx_window: int, gamma: int,
                            K: int, window_size: int, horizon: int = 36):
    """
    For each horizon h, constructs overlapping multiperiod returns and uses PCA to extract the K 
    factors. Computes mean variancce efficient weights and applies it to the OOS one-period return. 
    Args:
        - X_log: characteristics-based portfolio tensor - dim: (window_size, horizon, N)
    Returns:
        - returns: OOS one-period return - dim: (scalar,)
    """
    X_log = X_log[:, :, 0]
    X_fit = jax.lax.dynamic_slice(X_log, start_indices=(idx_window, 0), slice_sizes=(window_size, horizon)) # dim: window_size, N
    X_multi_fit = get_multiperiod_returns(X_fit, horizon, window_size) # dim: (window_size - horizon + 1, N), no wasted observations
    
    factors_pca, loadings_pca = RPPCA(X_multi_fit, gamma=gamma, K=K) # dim: (T, K) and (N, K)
    mu_pca = jnp.mean(factors_pca, axis=0) # dim: (K,)
    var_pca = newey_west_cov_jax(factors_pca, num_overlap=horizon-1) # dim: (K, K)
    mv_pca = jnp.linalg.inv(var_pca) @ mu_pca # dim: (K,)
    
    X_next = jax.lax.dynamic_slice(X_log, start_indices=(idx_window + window_size, 0, 0), slice_sizes=(horizon, horizon, N)) # dim: 1, N
    X_multi_next = get_multiperiod_returns(X_next, horizon, horizon)[0] # dim: (N,)

    factors_next_pca = X_multi_next @ loadings_pca @ jnp.linalg.inv(loadings_pca.T @ loadings_pca) # dim: (1, K)
    ret_pca = factors_next_pca @ mv_pca # dim: scalar
    return ret_pca


@partial(jax.jit, backend=main_compute_device, static_argnums=(2, 3, 4, 5, 6,))
def Pooled_PCA_Multiperiod_Unmapped(X_log: jnp.ndarray, 
                            idx_window: int, 
                            K: int, 
                            window_size: int,
                            lag: int,
                            max_horizon: int = 36,
                            gamma: int = -1):
    """
    This version applies PCA to a matrix (window_size, lag * N), so applies PCA to matrix of portfolios
    formed using lagged signals. Obtains factors (window_size, K) and loadings (lag * N, K). 
    Args:
        - X_log: (T, lag, N)
    Returns:
    """
    N = X_log.shape[2]
    X_fit = jax.lax.dynamic_slice(X_log, start_indices=(idx_window, 0, 0), slice_sizes=(window_size, lag, N)).reshape(window_size, -1) # dim: window_size, N * lag

    factors_pca, loadings_pca = RPPCA(X=X_fit, gamma=gamma, K=K) # dim: (window_size, K) and (lag * N, K)
    
    if K > 1:
        mu_pca = jnp.mean(factors_pca, axis=0) # dim: (K,)
        var_pca = jnp.cov(factors_pca.T) # dim: (K, K)
        mv_pca = jnp.linalg.inv(var_pca) @ mu_pca # dim: (K,)
    else:
        mv_pca = jnp.expand_dims(jnp.mean(factors_pca) / jnp.var(factors_pca), axis=-1)
    
    X_next = jax.lax.dynamic_slice(X_log, start_indices=(idx_window + window_size, 0, 0), slice_sizes=(max_horizon, lag, N)).reshape(max_horizon, -1) # dim: max_horizon, N * lag
    factors_next_pca = X_next @ loadings_pca @ jnp.linalg.inv(loadings_pca.T @ loadings_pca) # dim: (max_horizon, K)
    ret_pca = jnp.cumsum(factors_next_pca @ mv_pca, axis=0)
    return ret_pca

    
def RPPCA_One_Window_Jax(X_fit: jnp.ndarray, X_oos: jnp.ndarray, lst_K: List[int], gamma: int = -1):
    """
    Generates the results in the one_fit folder    
    """
    T_fit, N = X_fit.shape
    T_oos = X_oos.shape[0]

    Fhat, Lambdahat = [jnp.real(Q) for Q in RPPCA(X=X_fit, K=max(lst_K), gamma=gamma)]

    @partial(jax.jit, backend=main_compute_device, static_argnums=(2,))
    def _process_one_k(Fhat: jnp.ndarray, Lambdahat: jnp.ndarray, k: int):
        if k > 1:
            SDFweights = jnp.linalg.inv(jnp.cov(Fhat.T)) @ jnp.mean(Fhat, axis=0).T
        else:  
            SDFweights = jnp.mean(Fhat) / jnp.var(Fhat)
            SDFweights = jnp.array([SDFweights])
        SDFweightsassets_k = Lambdahat @ SDFweights # dim: (N,)

        # Get oos ret
        ret_oos = X_oos @ SDFweightsassets_k
        sr_oos = jnp.mean(ret_oos) / jnp.std(ret_oos)
        F_oos_fitted = X_oos @ Lambdahat # estimated factor values for OOS period?
        X_fitted_oos = F_oos_fitted @ Lambdahat.T
        return { 
            'ret_mv_oos': ret_oos,
            'X_fitted_oos': X_fitted_oos,
            'sr_oos': sr_oos
        }
    
    res = {}
    pbar = tqdm(total=len(lst_K))
    for K in lst_K:
        res[K] = _process_one_k(Fhat[:, :K], Lambdahat[:, :K], K)
        pbar.update(1)
    return res

def RPPCA_One_Window(X_fit: jnp.ndarray, X_oos: jnp.ndarray, lst_K: List[int], gamma: int = -1):
    T_fit, N = X_fit.shape
    T_oos = X_oos.shape[0]

    mat_ret_mv_oos = jnp.zeros((T_oos, len(lst_K))) # OOS ret or rx 
    mat_X_oos_fitted = jnp.zeros((T_oos, len(lst_K), N))

    Fhat, Lambdahat = [jnp.real(Q) for Q in RPPCA(X=X_fit, K=max(lst_K), gamma=gamma)]
    for idx_K, K in enumerate(lst_K):

        ### get MV ptf oos return - could write this into a helper function, not jit'ed
        if K > 1:
            SDFweights = jnp.linalg.inv(jnp.cov(Fhat[:, :K].T)) @ jnp.mean(Fhat[:, :K], axis=0).T
        else:  
            SDFweights = jnp.mean(Fhat[:, :K]) / jnp.var(Fhat[:, :K])
            SDFweights = jnp.array([SDFweights])
        # Can move below logic into helper function - args are lambdahat[:, :K], SDFweights, X_oos
        # Returns slice of mat_ret_mv_oos and mat_X_oos_fitted
        SDFweightsassets_k = Lambdahat[:, :K] @ SDFweights # dim: (N,)

        # Get oos ret
        ret_oos = X_oos @ SDFweightsassets_k
        mat_ret_mv_oos = mat_ret_mv_oos.at[:, idx_K].set(ret_oos)
        
        ### get fit of X_oos
        F_oos_fitted = X_oos @ Lambdahat[:, :K] # estimated factor values for OOS period?
    
        X_oos_fitted = F_oos_fitted @ Lambdahat[:, :K].T # dim: (TxK) * (NxK)^T = TxN
        mat_X_oos_fitted = mat_X_oos_fitted.at[:, idx_K, :].set(X_oos_fitted)
        
    sr_oos = mat_ret_mv_oos.mean(axis=0) / mat_ret_mv_oos.std(axis=0)
    dict_out = {
        'mat_X_fitted_oos': mat_X_oos_fitted, # dim: (T_oos, len(lst_K), N) - multi-horizon return predictions for all factors, times
        'mat_ret_mv_oos': mat_ret_mv_oos, # dim: (T_oos, len(lst_K)) - oos returns computed using is sdf weights
        'sr_oos': sr_oos, # dim: (len(lst_K)) - oos sharpe ratio
    }
    return dict_out


# def SDF_construct(Fhat, Lambdahat, lst_K = None, use_newey_west_cov = False, num_overlap_nw = 0):
#     """ 
#     - Construct SDF and mean-variance portfolio given factors and loadings
#     Args (numpy arrays):
#         - Fhat (T, K): factor estimates
#         - Lambdahat (N, K): loading estiamtes
#     Returns (numpy arrays):
#         - mat_SDF (T, K): SDF by successively including factors
#         - mat_SDFweightsassets (N, K): SDF weights on input assets/portfolios
#     """ 
#     N, K = Lambdahat.shape
#     T = Fhat.shape[0]
#     # By default use all PCs
#     if lst_K is None:
#         lst_K = jnp.arange(1, 1 + K)
#     if use_newey_west_cov:
#         from ..utils import newey_west_cov
    
#     # Include the first k factors for SDF construction
#     mat_SDF = jnp.full((T, len(lst_K)), jnp.nan)
#     mat_SDFweightsassets = jnp.full((N, len(lst_K)), jnp.nan)
#     for ind_k, k in enumerate(lst_K):
#         # Mean variance optimization, where SDFweights has dim (k,)
#         factors_k = Fhat[:, :k]
#         if k > 1:
#             if use_newey_west_cov:
#                 cov = newey_west_cov(factors_k, num_overlap=num_overlap_nw)
#             else:
#                 cov = jnp.cov(factors_k.T)
#             SDFweights = jnp.linalg.inv(cov) @ jnp.mean(factors_k, axis=0).T
#         else:
#             if use_newey_west_cov:
#                 cov = newey_west_cov(factors_k, num_overlap=num_overlap_nw)
#             else:
#                 cov = jnp.var(factors_k)
#             SDFweights = jnp.mean(factors_k) / jnp.var(factors_k)
#             SDFweights = jnp.array([SDFweights])

#         SDF_k = factors_k @ SDFweights # dim: (T,)
#         SDFweightsassets_k = Lambdahat[:, :k] @ jnp.linalg.inv(Lambdahat[:, :k].T @ Lambdahat[:, :k]) @ SDFweights # dim: (N,)

#         mat_SDF[:, ind_k] = SDF_k
#         mat_SDFweightsassets[:, ind_k] = SDFweightsassets_k
#     return mat_SDF, mat_SDFweightsassets


# def RPPCA_rolling_window(X, K, window_size,gamma=10):
#     """ 
#     - Run RPPCA with a rolling window
#     - Options for variancenormalization, orthogonalization, stdnorm are set to False
#     - num_window = T-window_size-1
#     Args:
#         - X is T by N
#         - K is max number of PC to extract
#         - RPPCA parameter gamma
#     Returns (numpy array): 
#         - mat_SDFweightsassets (num_window, N, K): weights on assets by combining the first k factors
#         - mat_ret_oos (num_window, K): OOS (excess) returns by combining the first k factors
#         - mat_gc (num_window,K): GC between constant and time-varying loadings
        
#     """ 
#     T, N = X.shape
#     num_window = T-window_size
#     assert K <= N
#     assert num_window > 0

#     # fit RPPCA on all data to get constant loading
#     _, W = RPPCA(X=X, K=K, gamma=gamma)

#     mat_SDFweightsassets = jnp.full((num_window,N,K),jnp.nan)
#     mat_ret_oos = jnp.full((num_window,K), jnp.nan) # OOS ret or rx 
#     mat_gc = jnp.full((num_window,K), jnp.nan)

#     for t in range(num_window):
#         X_fit = X[t:(t+window_size), :]
#         X_oos = X[t+window_size, :]

#         # fit RPPCA
#         Fhat,Lambdahat = RPPCA(X=X_fit,K=K,gamma=gamma)
#         # normalize signs of the loadings to align factors over time
#         # this step assumes factor ordering doesn't change
#         if t > 0:
#             sign_temp=jnp.sign(jnp.diag(Lambdahat.T@Lambdahatprevious))
#             Lambdahat=Lambdahat@jnp.diag(sign_temp)
#             Fhat=Fhat@jnp.diag(sign_temp)
#         Lambdahatprevious=Lambdahat
        
#         # construct SDF and get OOS return
#         mat_SDF_t, mat_SDFweightsassets_t = SDF_construct(Fhat=Fhat,Lambdahat=Lambdahat)
#         ret_oos = X_oos @ mat_SDFweightsassets_t

#         # GC with constant loading
#         gc,_=GC(W=W,Lambdahat=Lambdahat)

#         # record results
#         mat_SDFweightsassets[t,:]=mat_SDFweightsassets_t
#         mat_ret_oos[t,:]=ret_oos
#         mat_gc[t,:]=gc
        
#     # return
#     return mat_SDFweightsassets, mat_ret_oos,mat_gc


# def RPPCA_rolling_window_lite(X,lst_K,window_size,gamma=10):
#     assert len(X.shape)==2
#     T,N=X.shape
#     num_window=T-window_size
#     assert num_window>0

#     mat_ret_mv_oos=jnp.full((num_window,len(lst_K)),jnp.nan) # OOS ret or rx 
#     mat_X_next_fitted=jnp.full((num_window, len(lst_K),N),jnp.nan)

#     pbar=tqdm(total=num_window)
#     for idx_window in range(num_window):
#         X_fit=X[idx_window:(idx_window+window_size),:]
#         X_next=X[idx_window+window_size,:]

#         # fit RPPCA
#         Fhat,Lambdahat=[jnp.real(Q) for Q in RPPCA(X=X_fit,K=max(lst_K),gamma=gamma)]

#         for idx_K, K in enumerate(lst_K):
#             ### get MV ptf oos return
#             if K>1:
#                 SDFweights=jnp.linalg.inv(jnp.cov(Fhat[:,:K].T))@jnp.mean(Fhat[:,:K],axis=0).T
#             else:
#                 SDFweights=jnp.mean(Fhat[:,:K])/jnp.var(Fhat[:,:K])
#                 SDFweights=jnp.array([SDFweights])
#             SDFweightsassets_k=Lambdahat[:,:K]@SDFweights # dim: (N,)
#             ret_oos=X_next@SDFweightsassets_k
#             mat_ret_mv_oos[idx_window,idx_K]=ret_oos

#             ### get fit of X_next
#             F_next_fitted=X_next@Lambdahat[:,:K]
#             X_next_fitted=Lambdahat[:,:K]@F_next_fitted
#             mat_X_next_fitted[idx_window,idx_K,:]=X_next_fitted
#         pbar.update(1)
        
#     dict_out={'mat_X_next_fitted':mat_X_next_fitted,'mat_ret_mv_oos':mat_ret_mv_oos}
#     return dict_out