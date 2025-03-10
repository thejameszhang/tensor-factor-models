import time
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
from typing import List, Tuple
from tfm.parafac_jax import parafac_enhanced, normalize_factors
from tfm.utils._constants import *

@partial(jax.jit, backend=main_compute_device, static_argnums=(1,))
def unfold(tensor: jnp.ndarray, mode: int) -> jnp.ndarray:
    """Matricize the tensor along the specified mode."""
    return jnp.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

@partial(jax.jit, backend=main_compute_device, static_argnums=(1, 2))
def initialize_factors_svd(tensor: jnp.ndarray, rank: int, random_state: int = 0) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Initialize factor matrices using SVD."""
    weights = jnp.ones(rank) # Initial weights are different then src, but doesn't change anything
    factors = []
    key = jax.random.PRNGKey(random_state)
    keys = jax.random.split(key, tensor.ndim)
    
    for k, mode in zip(keys, range(tensor.ndim)):
        matricized = unfold(tensor, mode)
        u, s, _ = jnp.linalg.svd(matricized, full_matrices=False) # dim: 

        if rank > u.shape[1]:
            shape = (u.shape[0], rank - u.shape[1])
            random_part = jax.random.uniform(k, shape, minval=0.0, maxval=1.0)
            u = jnp.concatenate([u, random_part], axis=1)

        factors.append(u[:, :rank])
    
    return weights, factors

@partial(jax.jit, backend=main_compute_device)
def khatri_rao_product(matrices: List[jnp.ndarray]) -> jnp.ndarray:
    """Compute the Khatri-Rao product of a list of matrices."""
    n_cols = matrices[0].shape[1]
    result = matrices[0]
    for matrix in matrices[1:]:
        result = jnp.einsum('ir,jr->ijr', result, matrix).reshape(-1, n_cols)
    return result


@partial(jax.jit, backend=main_compute_device, static_argnums=(1,))
def khatri_rao(matrices, mode):
    """
    Compute the Khatri-Rao product of a list of matrices with the functionality of skipping a mode. 
    """
    selected = [mat for i, mat in enumerate(matrices) if i != mode]
    return khatri_rao_product(selected)


@partial(jax.jit, backend=main_compute_device, static_argnums=(1))
def soft_threshold(X, threshold):
    """Elementwise soft-thresholding operator."""
    return jnp.sign(X) * jnp.maximum(jnp.abs(X) - threshold, 0)

@partial(jax.jit, backend=main_compute_device, static_argnums=(2,))
def unfolding_dot_khatri_rao(tensor, factors, mode):
    kr_factors = khatri_rao(factors, mode)
    mttkrp = jnp.dot(unfold(tensor, mode), jnp.conj(kr_factors))
    return mttkrp

# @partial(jax.jit, backend=main_compute_device, static_argnums=(1, 2, 3, 4, 5, 6))
# def admm_update(UtM, UtU, x, dual_var, rho, l1_reg, tol, max_iter):

#     def cond(state):
#         x, x_split, dual_var, iteration, _ = state
#         dual_residual = x - x_split
#         primal_residual = x - state[4]  # x_old
#         return (iteration < max_iter) & (
#             (jnp.linalg.norm(dual_residual) >= tol * jnp.linalg.norm(x)) |
#             (jnp.linalg.norm(primal_residual) >= tol * jnp.linalg.norm(dual_var))
#         )
    
#     def body(state):
#         x, x_split, dual_var, iteration, x_old = state
        
#         # x_split update
#         x_split = jnp.linalg.solve(
#             (UtU + rho * jnp.eye(UtU.shape[0])).T,
#             (UtM + rho * (x + dual_var)).T
#         )
        
#         # x update with L1 regularization
#         x_new = soft_threshold(x_split.T - dual_var, l1_reg / rho)
        
#         # Dual variable update
#         dual_var = dual_var + x_new - x_split.T
        
#         return (x_new, x_split.T, dual_var, iteration + 1, x)
    
#     x_split = x.copy()
#     state = (x, x_split, dual_var, 0, x)
#     x, x_split, dual_var, _, _ = jax.lax.while_loop(cond, body, state)
    
#     return x, x_split, dual_var

# ---------------------------
# ADMM-based PARAFAC (JAX)
# ---------------------------
@partial(jax.jit, backend=main_compute_device, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def parafac_admm(tensor: jnp.ndarray, 
                 rank: int, 
                 l1_reg: Tuple = (0, 0, 0), 
                 max_iter: int = 50, 
                 admm_rho: float = 1.0, 
                 admm_tol: float = 1e-6,
                 admm_max_iter: int = 10, 
                 init: str = "svd"):
    """
    Compute the PARAFAC (CP) decomposition of a tensor with L1 regularization via ADMM.
    Issue with early stopping here due to jax.lax.scan limitations. 
    Minimizes:
         0.5 * ||X - [A^(1), ..., A^(N)]||_F^2 + l1_reg * sum ||A^(n)||_1
    using ADMM to handle the L1 term.
    """
    dims = tensor.shape
    n_modes = len(dims)
    
    if init == "svd":
        _, factors = initialize_factors_svd(tensor, rank)
    # else:
    #     key = random.PRNGKey(random_key)
    #     pass

    # Dual variables for ADMM per factor
    duals = []
    factors_aux = []
    for i in range(len(factors)):
        duals.append(jnp.zeros_like(factors[i]))
        factors_aux.append(jnp.zeros_like(factors[i].T))

    weights = jnp.ones(rank)

    
    for _ in range(max_iter):
        for mode in range(n_modes):
            threshold = l1_reg[mode] / admm_rho
            Xn = unfold(tensor, mode)
            W = khatri_rao(factors, mode)
            WtW = W.T @ W
            XnW = Xn @ W

            # ADMM variables for current mode update
            A = factors[mode]
            Z = A.copy()
            U = duals[mode]
            
            for _ in range(admm_max_iter):
                A_new = (XnW + admm_rho * (Z - U)) @ jnp.linalg.inv(WtW + admm_rho * jnp.eye(rank))
                Z_new = soft_threshold(A_new + U, threshold)
                U_new = U + A_new - Z_new
                
                # if jnp.linalg.norm(A_new - Z_new) < admm_tol:
                #     A, Z, U = A_new, Z_new, U_new
                #     break
                A, Z, U = A_new, Z_new, U_new

            factors[mode] = A
            duals[mode] = U
        
        # diff = sum(jnp.linalg.norm(f - pf) for f, pf in zip(factors, prev_factors))
        # if diff < tol:
        #     break
    return weights, factors


@partial(jax.jit, backend=main_compute_device)
def reconstruct(weights: jnp.ndarray, factors: List[jnp.ndarray]) -> jnp.ndarray:
    """Reconstruct tensor from weights and factors."""
    rank = weights.shape[0]
    reconstruction = jnp.zeros(tuple(f.shape[0] for f in factors))
    
    for r in range(rank):
        # Get rank-1 components
        components = [f[:, r] for f in factors]
        # Compute outer product
        rank_one = components[0]
        for c in components[1:]:
            rank_one = rank_one.reshape(rank_one.shape + (1,)) * c
        reconstruction += weights[r] * rank_one
    return reconstruction

# ---------------------------
# Benchmark: Compare ADMM vs. TensorLy
# ---------------------------
def benchmark_comparison():
    """
    Run both the ADMM-based parafac and TensorLy's constrained_parafac on the same synthetic tensor.
    Print the relative reconstruction errors for each.
    """
    print("Running benchmark: ADMM vs. ALS\n")
    ITERATIONS = 100
    X = jnp.load(f'../organized_data/organized_data/scs/mat_ptf_re_lag_120.npz')['mat_ptf_re_rank']
    T = 120
    L = 36
    X_log = jnp.log(1 + X)[:T, :L]
    rank = 20
    
    l1_reg = (0, 1e-3, 0)
    
    # get things warmed up for jit
    parafac_admm(X_log, rank, l1_reg)
    parafac_enhanced(X_log, rank)

    time_start = time.time()
    for _ in range(ITERATIONS):
        weights_admm, factors_admm = parafac_admm(X_log, rank, l1_reg)
    time_end = time.time()
    print(time_end - time_start)

    X_est_admm = reconstruct(weights_admm, factors_admm)
    error_admm = jnp.linalg.norm(X_log - X_est_admm) / jnp.linalg.norm(X_log)
        
    
    time_start = time.time()
    for _ in range(ITERATIONS):
        weights_als, factors_als = parafac_enhanced(X_log, rank)
    time_end = time.time()
    print(time_end - time_start)
    X_als = reconstruct(weights_als, factors_als)
    error_als = jnp.linalg.norm(X_log - X_als) / jnp.linalg.norm(X_log)
    
    print(f"ADMM-based parafac relative reconstruction error: {error_admm:.10f}")
    print(f"ALS-based parafac relative reconstruction error: {error_als:.10f}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    os.environ['CUDA_VISIBLE_DEVICES'] = "1" # Titan and FP64 mode
    jax.config.update('jax_platform_name', 'gpu')
    print("JAX is using device:", jax.devices()[0], jax.devices())
    benchmark_comparison()
