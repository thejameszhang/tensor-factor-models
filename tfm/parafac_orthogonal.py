import jax
import jax.numpy as jnp
from typing import Tuple, List
from functools import partial
from tfm.utils._constants import *
from tfm.parafac_jax import (
    initialize_factors_svd,
    initialize_factors_random,
    matricize,
    khatri_rao_product,
    reconstruct_tensor,
    update_factor_fixed_intercept
)

# @partial(jax.jit, backend=main_compute_device, static_argnums=(2))
# def update_orthonormal_factor(tensor: jnp.ndarray, factors: List[jnp.ndarray], mode: int, weights: jnp.ndarray) -> jnp.ndarray:
#     """Update factor matrix with orthogonality constraint using SVD. Orthonormal constraint. """
#     other_factors = [f for i, f in enumerate(factors) if i != mode]
#     kr_product = khatri_rao_product(other_factors)
    
#     # Weight the Khatri-Rao product
#     weighted_kr = kr_product * weights[None, :]
    
#     # Matricize the tensor
#     unfolded = matricize(tensor, mode)
    
#     # Compute the matrix for SVD
#     matrix = unfolded @ weighted_kr
    
#     # Use SVD to find the orthogonal factor matrix
#     # U, _, Vh = jnp.linalg.svd(matrix, full_matrices=False)
#     # updated_factor = U @ Vh
#     updated_factor, _ = jnp.linalg.qr(matrix)
#     return updated_factor


@partial(jax.jit, backend=main_compute_device, static_argnums=(2))
def update_orthogonal_factor(tensor: jnp.ndarray, factors: List[jnp.ndarray], mode: int, weights: jnp.ndarray) -> jnp.ndarray:
    """Update factor matrix with orthogonality constraint using:
       A = XZ' @ (ZX'XZ')^{-0.5}, computed via SVD for GPU efficiency.
       PARAFAC. Tutorial and applications by Rasmus Bro, Section 4.3"""
    # Exclude the factor being updated
    other_factors = [f for i, f in enumerate(factors) if i != mode]
    
    # Compute Khatri-Rao product and apply weights
    kr_product = khatri_rao_product(other_factors)
    weighted_kr = kr_product * weights[None, :]
    
    unfolded = matricize(tensor, mode)  # X in the formula
    X = unfolded
    Z = weighted_kr.T

    XZt = X @ Z.T  # Shape: (I_mode, R)
    ZXXZt = Z @ X.T @ X @ Z.T  # Shape: (R, R)

    U, S, Vh = jnp.linalg.svd(ZXXZt, full_matrices=False, hermitian=True) # check this
    inv_sqrt = U @ jnp.diag(S**-0.5) @ Vh

    updated_factor = XZt @ inv_sqrt
    return updated_factor



@partial(jax.jit, backend=main_compute_device, static_argnums=(1, 2, 3, 4, 5, 6))
def parafac_orthogonal(
    tensor: jnp.ndarray,
    rank: int,
    orthogonal_mode: int = 0, # factors
    fixed_intercept_mode: int = 1, # lag loadings
    random_state: int = 0,
    n_iter_max: int = 100,
    tol: float = 1e-8,
    init: str = 'svd'
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """PARAFAC decomposition with orthogonality constraint in one mode and also fixed intercept constraints.
    
    Args:
        tensor: Input tensor to decompose
        rank: Number of components
        orthogonal_mode: Mode in which to enforce orthogonality (0-based indexing)
        fixed_intercept_mode: Mode in which lag loadings have value 1 prior to normalization
        random_state: Random seed for initialization
        n_iter_max: Maximum number of iterations
        tol: Convergence tolerance
        init: Initialization method ('svd' or 'random')
    
    Returns:
        weights: Component weights
        factors: List of factor matrices
    """
    # Initialize factors
    if init == 'svd':
        weights, factors = initialize_factors_svd(tensor, rank, random_state)
    else:
        weights, factors = initialize_factors_random(tensor, rank, random_state)
    
    # Enforce initial orthogonality in specified mode
    U, _, Vh = jnp.linalg.svd(factors[orthogonal_mode], full_matrices=False)
    factors[orthogonal_mode] = U @ Vh
    
    def body_fun(carry, _):
        factors, weights, prev_error = carry
        
        # Store previous reconstruction for convergence check
        prev_reconstruction = reconstruct_tensor(weights, factors)
        
        # Update each factor matrix
        for mode in range(tensor.ndim):
            # Orthogonal mode - time series factors
            if mode == orthogonal_mode:
                new_factor = update_orthogonal_factor(tensor, factors, mode, weights) 
            # Fixed intercept mode - lag loadings
            elif mode == fixed_intercept_mode:
                new_factor, one_lag_weights = update_factor_fixed_intercept(tensor, factors, mode, weights)
            # Normal mode - cross-sectional loadings
            else:
                new_factor = update_factor(tensor, factors, mode, weights)
            
            factors = [new_factor if i == mode else f for i, f in enumerate(factors)]
            if mode == fixed_intercept_mode:
                factors[0] = factors[0] * (one_lag_weights[None, :] + 1e-12)

        
        # Calculate norms of each factor
        norms = [jnp.linalg.norm(f, axis=0) for f in factors]
        
        # Update weights with product of norms
        weights = weights * jnp.prod(jnp.stack(norms), axis=0)
        
        # Normalize factors
        factors = [
            f / (norm[None, :] + 1e-12)
            for f, norm in zip(factors, norms)
        ]
        
        # Calculate error using relative change in reconstruction
        new_reconstruction = reconstruct_tensor(weights, factors)
        error = jnp.linalg.norm(new_reconstruction - prev_reconstruction) / jnp.linalg.norm(prev_reconstruction)
        
        return (factors, weights, error), error
    
    # Run iterations
    init_carry = (factors, weights, jnp.inf)
    (factors, weights, _), errors = jax.lax.scan(body_fun, init_carry, None, length=n_iter_max)
    
    return weights, factors


@partial(jax.jit, backend=main_compute_device, static_argnums=(2))
def update_factor(tensor: jnp.ndarray, factors: List[jnp.ndarray], mode: int, weights: jnp.ndarray) -> jnp.ndarray:
    """Update factor matrix using ALS with improved numerical stability."""
    other_factors = [f for i, f in enumerate(factors) if i != mode]
    kr_product = khatri_rao_product(other_factors)
    
    # Weight the Khatri-Rao product
    weighted_kr = kr_product * weights[None, :]
    
    # Matricize the tensor
    unfolded = matricize(tensor, mode)
    
    # Compute normal equations with regularization
    gram = weighted_kr.T @ weighted_kr
    rhs = unfolded @ weighted_kr
    
    # Add regularization scaled to the matrix norm
    reg_scale = 1e-10 * jnp.linalg.norm(gram)
    reg = jnp.eye(gram.shape[0]) * reg_scale
    
    # Solve the regularized system
    updated_factor = jnp.linalg.solve(gram + reg, rhs.T).T
    return updated_factor

# @partial(jax.jit, backend=main_compute_device, static_argnums=(1, 2, 3))
def orth_als(tensor: jnp.ndarray, rank: int, n_iter_max: int = 100, 
             random_state: int = 0) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Orthogonal-ALS algorithm for tensor decomposition.
    
    Args:
        tensor: Input tensor to decompose
        rank: Number of components
        n_iter_max: Maximum number of iterations
        random_state: Random seed for initialization
    
    Returns:
        weights: Component weights
        factors: List of factor matrices
    """
    # Initialize factors
    weights, factors = initialize_factors_svd(tensor, rank, random_state)

    def body_fun(carry, _):
        factors, weights = carry
    # for i in range(n_iter_max):
        factors = [jnp.linalg.qr(M)[0] for M in factors]
        new_factors = []
        
        for mode in range(tensor.ndim):
            other_factors = [f for i, f in enumerate(factors) if i != mode]
            kr_product = khatri_rao_product(other_factors)
            
            # Weight the Khatri-Rao product
            weighted_kr = kr_product * weights[None, :]
            
            # Matricize the tensor
            unfolded = matricize(tensor, mode)
            matrix = unfolded @ weighted_kr
            new_factors.append(matrix)

        # Calculate norms of each factor
        # norms = [jnp.linalg.norm(f, axis=0) for f in new_factors]
        # # print(norms)
        
        # # Update weights with product of norms
        # weights = weights * jnp.prod(jnp.stack(norms), axis=0)
        
        # # Normalize factors
        # factors = [
        #     f / (norm[None, :] + 1e-12)
        #     for f, norm in zip(factors, norms)
        # ]

        return (factors, weights), None

    init_carry = (factors, weights)
    (factors, weights), _ = jax.lax.scan(body_fun, init_carry, None, length=n_iter_max)
    return weights, factors