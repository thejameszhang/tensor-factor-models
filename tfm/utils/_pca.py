from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from typing import List
from tfm.utils._cov import newey_west_cov_jax
from tfm.utils._constants import *

@partial(jax.jit, backend=main_compute_device, static_argnums=(2,))
def RPPCA(X: jnp.ndarray, gamma: int, K: int):
    """ 
    - Run RPPCA at a single time step
    - Options for variancenormalization, orthogonalization, stdnorm are set to False.
    - Gamma=-1 is standard PCA
    Args:
        - X is T by N
        - K is max number of PC to extract
        - RPPCA parameter gamma
    Returns: 
        - Fhat (T, K)
        - Lambdahat (N, K) 
    """ 
    T, N = X.shape
    # assert K <= N
    WN = jnp.eye(N)
    WT = jnp.eye(T) + gamma * jnp.ones((T,T)) / T

    # Generic estimator for general weighting matrices
    Xtilde = X @ WN
    # Covariance matrix with weighted mean
    VarWPCA = Xtilde.T @ WT @ Xtilde / T 

    # Eigenvalue decomposition
    DWPCA, VWPCA = jnp.linalg.eigh(VarWPCA) # bottleneck for PCA, use eigh faster for cov matrix
    DWPCA, VWPCA = DWPCA.astype(dtype_compact), VWPCA.astype(dtype_compact)
    order = jnp.argsort(DWPCA)[::-1]
    DWPCA = DWPCA[order]
    VWPCA = VWPCA[:, order]

    # Lambdahat are the eigenvectors after reverting the cross-sectional transformation
    Lambdahat = VWPCA[:, :K]
    # Normalizing the signs of the loadings
    Fhat = X @ Lambdahat @ jnp.linalg.inv(Lambdahat.T @ Lambdahat)
    sign_temp = jnp.sign(Fhat.mean(axis=0))
    Lambdahat = Lambdahat @ jnp.diag(sign_temp)
    Fhat = Fhat @ jnp.diag(sign_temp)
    return jnp.real(Fhat), jnp.real(Lambdahat)



@partial(jax.jit, backend=main_compute_device, static_argnums=(1, 2))
def get_multiperiod_returns(X: jnp.ndarray, horizon: int, window_size: int):
    """
    """
    @partial(jax.jit, backend=main_compute_device, static_argnums=(1,))
    def _process(t: int, horizon: int):
        # Original code
        # X_multi_s_temp = jnp.zeros(num_ptf)
        # for l in range(horizon):
        #     X_multi_s_temp += X[t+l, l, :]

        @partial(jax.jit, backend=main_compute_device)
        def _cumulative(l: int):
            return X[t+l, l, :] # is this indexing correct?, it could be instead X[t-l, s-l-1] (10, 4), (9, 3), (8, 2), (7, 1), (6, 0)

        return jnp.sum(jax.vmap(_cumulative)(jnp.arange(horizon)), axis=0) # dim: (num_ptf)
        
    X_multi = jax.vmap(_process, in_axes=(0, None))(jnp.arange(window_size - horizon + 1), horizon)
    # X_multi = jax.vmap(_process, in_axes=(0, None))(jnp.arange(horizon - 1, window_size), horizon)
    return X_multi # dim: (window_size - horizon + 1, num_ptf)


# @partial(jax.jit, backend=main_compute_device)
# def get_F_next(X_next: jnp.ndarray, loadings: jnp.ndarray, idx_h: int):
#     loadings_h = loadings[idx_h]
#     print("loadings", loadings_h.shape)
#     return (X_next[idx_h] @ loadings_h @ jnp.linalg.inv(loadings_h.T @ loadings_h)).cumsum(axis=0) # dim: (max_horizon, K)

@partial(jax.jit, backend=main_compute_device)
def get_factor_means(F: jnp.ndarray, idx_h: int):
    return jnp.mean(F[:, idx_h, :], axis=0)

@partial(jax.jit, backend=main_compute_device, static_argnums=(2))
def get_mv_weights(mu: jnp.ndarray, cov: jnp.ndarray, K: int, idx_h: int):
    if K > 1:
        return jnp.linalg.inv(cov[idx_h]) @ mu[idx_h]
    else:
        return jnp.mean(mu) / jnp.var(cov)

def SDF_construct(Fhat,Lambdahat, lst_K=None, use_newey_west_cov = False, num_overlap_nw = 0):
    """
    Construct SDF and mean-variance portfolio given factors and loadings
    Args (numpy arrays):
        Fhat (T,K): factor estimates
        Lambdahat (N,K): loading estiamtes
    Returns (numpy arrays):
        mat_SDF (T, K): SDF by successively including factors
        mat_SDFweightsassets (N, K): SDF weights on input assets/portfolios
    """
    N,K=Lambdahat.shape
    T=Fhat.shape[0]
    if lst_K is None:
        lst_K=np.arange(1,1+K)
    if use_newey_west_cov:
        from tfm.utils._cov import newey_west_cov
    
    # include the first k factors for SDF construction
    mat_SDF=np.full((T,len(lst_K)),np.nan)
    mat_SDFweightsassets=np.full((N,len(lst_K)),np.nan)
    for ind_k,k in enumerate(lst_K):
        factors_k=Fhat[:,:k]
        # Mean variance optimization
        # SDFweights has dim (k,)
        if k>1:
            if use_newey_west_cov:
                cov=newey_west_cov(factors_k, num_overlap=num_overlap_nw)
            else:
                cov=np.cov(factors_k.T)
            SDFweights=np.linalg.inv(cov)@np.mean(factors_k,axis=0).T
        else:
            if use_newey_west_cov:
                cov=newey_west_cov(factors_k, num_overlap=num_overlap_nw)
            else:
                cov=np.var(factors_k)
            SDFweights=np.mean(factors_k)/np.var(factors_k)
            SDFweights=np.array([SDFweights])

        SDF_k=factors_k@SDFweights # dim: (T,)
        SDFweightsassets_k=Lambdahat[:,:k]@np.linalg.inv(Lambdahat[:,:k].T@Lambdahat[:,:k])@SDFweights # dim: (N,)

        mat_SDF[:,ind_k]=SDF_k
        mat_SDFweightsassets[:,ind_k]=SDFweightsassets_k
    return mat_SDF, mat_SDFweightsassets

