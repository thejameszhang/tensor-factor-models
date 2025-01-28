"""
Newey West Covariance Matrix Estimator
================================ 
- Turns out that converting to Numpy array and using Numpy implementation
is faster than JIT'ing -> long compilation time. 
================================
Author: James Zhang
Date: November 2024
"""
import numpy as np
import jax
import jax.numpy as jnp
import time
from tfm.utils._constants import *

def newey_west_cov(X, num_overlap=0):
    """
    Args:
        - X: T by N matrix of data, not centered
        - lag: number of overlapping periods. 
            num_overlap=0 gives back X_center.T@X_center/T
    Returns:
        - cov: newey-west cov estimator
    """
    
    if num_overlap < 0:
        raise ValueError('num_overlap should be >=0')

    T, N = X.shape
    X_center = X - X.mean(axis=0)
    cov = X_center.T @ X_center / T
    for j in range(1, 1 + num_overlap):
        w_j = 1 - (j / (num_overlap + 1))
        cov_j_temp = X_center[j:, :].T @ X_center[:-j, :] / T
        cov_j = cov_j_temp + cov_j_temp.T
        cov += w_j * cov_j
    return cov
    

@partial(jax.jit, backend=main_compute_device, static_argnums=(1,))
def newey_west_cov_jax(X: jnp.ndarray, num_overlap: int):
    """
    Args:
        - X: T by N matrix of data, not centered
        - num_overlap: number of overlapping periods. 
            num_overlap=0 gives back X_center.T@X_center/T
    Returns:
        - cov: newey-west cov estimator
    """

    T, N = X.shape
    X_center = X - X.mean(axis=0)
    cov = X_center.T @ X_center / T
    for j in range(1, 1 + num_overlap):
        w_j = 1 - (j / (num_overlap + 1))
        cov_j_temp = X_center[j:, :].T @ X_center[:-j, :] / T
        cov_j = cov_j_temp + cov_j_temp.T
        cov += w_j * cov_j
    
    return cov


@partial(jax.jit, backend=main_compute_device, static_argnums=(1,))
def newey_west_cov_jax2(X: jnp.ndarray, num_overlap: int = 0):
    """
    Args:
        - X: T by N matrix of data, not centered
        - lag: number of overlapping periods. 
            num_overlap=0 gives back X_center.T@X_center/T
    Returns:
        - cov: newey-west cov estimator

    Hard to VMAP across j's -> dynamic shapes unless pre-compute *_temp. 
    """
    # if num_overlap < 0:
    #     raise ValueError('num_overlap should be >=0')
    
    T, _ = X.shape
    X_center = X - X.mean(axis=0)
    cov = X_center.T @ X_center / T
    # if num_overlap == 0:
    #     return cov[jnp.newaxis, :]
    w_i = jnp.array([(1 - (i / (num_overlap + 1))) for i in range(1, 1 + num_overlap)])[:, jnp.newaxis, jnp.newaxis]
    cov_i = jnp.array([X_center[i:, :].T @ X_center[:-i, :] / T for i in range(1, 1 + num_overlap)])
    outer = cov_i + jnp.transpose(cov_i, axes=(0, 2, 1))
    return jnp.sum(w_i * outer, axis=0) + jnp.expand_dims(cov, axis=0)


def orig_vs_jax():


    # Timing the optimized JAX implementation
    
    cov_jax = newey_west_cov_jax1(X_jax, num_overlap)
    cov_jax2 = newey_west_cov_jax2(X_jax, num_overlap)
    print(type(cov_jax), type(cov_jax2))

    start_time = time.time()
    for _ in range(1000):
        cov_jax = newey_west_cov_jax1(X_jax, num_overlap)
    jax_time = time.time() - start_time
    
    start_time2 = time.time()
    for _ in range(1000):
        cov_jax2 = newey_west_cov_jax2(X_jax, num_overlap)
    jax_time2 = time.time() - start_time2


    # Timing the original implementation
    # start_time = time.time()
    # X_numpy = np.asarray(X_jax)
    # for _ in range(100):
    #     cov_original = newey_west_cov(X_numpy, num_overlap)
    # original_time = time.time() - start_time

    # Compare the results
    print("Original Implementation Covariance Matrix:")
    print(cov_jax)

    print("\nOptimized JAX Implementation Covariance Matrix:")
    print(cov_jax2)

    # Check if the results are close
    if jnp.allclose(cov_jax, cov_jax2, atol=1e-6):
        print("\nThe results are consistent between the two implementations.")
    else:
        print("\nThe results differ between the two implementations.")

    # Print timing results
    print(f"\nTime taken by original implementation over 100 iterations: {jax_time:.4f} seconds")
    print(f"Time taken by JAX implementation over 100 iterations: {jax_time2:.4f} seconds")


if __name__ == "__main__":

    # Generate random data
    T, N = 100, 41 # Example dimensions
    X = np.random.randn(T, N)

    # Convert to JAX array for the JAX implementation
    X_jax = jnp.array(X)

    # Number of overlaps
    num_overlap = 4

    orig_vs_jax()



