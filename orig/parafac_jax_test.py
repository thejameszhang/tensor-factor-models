import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_enable_x64', True)
from functools import partial
from typing import List, Dict, Tuple, Optional
from code.code_11092024.orig.parafac_jax import *


@partial(jax.jit, static_argnums=(1,))
def normalize_factors(factors: Dict[str, jnp.ndarray], reorder: bool = True) -> Dict[str, jnp.ndarray]:
    """Normalize tensor factors and optionally reorder by explained variance."""
    F, W, B = factors['F'], factors['W'], factors['B']
    S = factors['S']
    
    # Normalize factors
    norms = {
        'F': jnp.linalg.norm(F, axis=0),
        'W': jnp.linalg.norm(W, axis=0),
        'B': jnp.linalg.norm(B, axis=0)
    }
    
    # Update weights and normalize factors
    new_S = S * norms['F'] * norms['W'] * norms['B']
    new_F = F / norms['F']
    new_W = W / norms['W']
    new_B = B / norms['B']
    
    if reorder:
        # Sort by explained variance
        idx_sort = jnp.argsort(-jnp.abs(new_S))
        new_S = new_S[idx_sort]
        new_F = new_F[:, idx_sort]
        new_W = new_W[:, idx_sort]
        new_B = new_B[:, idx_sort]
    
    return {
        'F': new_F,
        'W': new_W,
        'B': new_B,
        'S': new_S
    }

@partial(jax.jit, static_argnums=(2, 3, 4))
def Tensor_One_Window(X_fit: jnp.ndarray, 
                     X_oos: jnp.ndarray, 
                     lst_K: Tuple[int, ...],
                     random_seed: int = 42,
                     n_iter_max: int = 100):
    """Performs tensor decomposition and out-of-sample predictions with batched operations."""
    T_oos, max_lag, num_ptf = X_oos.shape
    
    @partial(jax.jit, static_argnums=(0,))  # Make K static
    def process_single_K(K: int, carry):
        """Process a single K value."""
        # Perform tensor decomposition
        weights, factors = parafac_enhanced(
            tensor=X_fit,
            rank=K,
            random_state=random_seed,
            n_iter_max=n_iter_max,
            fix_intercept_mode=0
        )
        
        # Extract and normalize factors
        factors = dict(zip(['F','W','B'], factors))
        factors['S'] = weights
        factors = normalize_factors(factors, reorder=True)
        
        # Compute Z_fit for all K components at once
        def compute_Z_row(i):
            return jnp.kron(factors['W'][:, i], factors['B'][:, i]) * factors['S'][i]
        
        Z_fit = jax.vmap(compute_Z_row)(jnp.arange(K))
        
        # Compute weights matrix once
        ZZt = Z_fit @ Z_fit.T
        mat_weight_flatten = jnp.linalg.solve(ZZt + 1e-10 * jnp.eye(K), Z_fit).T
        
        # Compute F_oos for all time points at once
        X_oos_flatten = X_oos.reshape(T_oos, -1)
        F_oos = X_oos_flatten @ mat_weight_flatten
        
        # Compute portfolio returns
        def compute_returns_k1():
            return F_oos * jnp.mean(factors['F']) / jnp.var(factors['F'])
            
        def compute_returns_kn():
            cov_inv = jnp.linalg.inv(jnp.cov(factors['F'].T) + 1e-10 * jnp.eye(K))
            mean_F = jnp.mean(factors['F'], axis=0)
            return F_oos @ cov_inv @ mean_F
            
        ret_mv_oos = jax.lax.cond(
            K == 1,
            compute_returns_k1,
            compute_returns_kn
        )
        
        # Compute Sharpe ratio
        sr_oos = jnp.mean(ret_mv_oos) / jnp.std(ret_mv_oos)
        
        # Compute fitted returns using batched operations
        def compute_BWS(i):
            return (factors['W'][:, i][:, None] @ 
                   factors['B'][:, i][None, :] * 
                   factors['S'][i])
            
        BWS = jax.vmap(compute_BWS)(jnp.arange(K))
        
        X_fitted_oos = jnp.sum(
            F_oos[:, :, None, None] * BWS[None, :, :, :],
            axis=1
        ).reshape(T_oos, max_lag, num_ptf)
        
        return carry, (ret_mv_oos, X_fitted_oos, sr_oos)
    
    def body_fun(i, val):
        carry, (ret_results, X_results, sr_results), K = val
        new_carry, (ret_mv_oos, X_fitted_oos, sr_oos) = process_single_K(K[i], carry)
        
        # Update each array separately
        new_ret_results = ret_results.at[i].set(ret_mv_oos)
        new_X_results = X_results.at[i].set(X_fitted_oos)
        new_sr_results = sr_results.at[i].set(sr_oos)
        
        return (new_carry, (new_ret_results, new_X_results, new_sr_results), K)
    
    # Initialize separate arrays for each result type
    init_ret_results = jnp.zeros((len(lst_K), T_oos))
    init_X_results = jnp.zeros((len(lst_K), T_oos, max_lag, num_ptf))
    init_sr_results = jnp.zeros(len(lst_K))
    
    # Run the loop
    init_carry = None
    _, (final_ret_results, final_X_results, final_sr_results) = jax.lax.fori_loop(
        0, len(lst_K), body_fun, (init_carry, (init_ret_results, init_X_results, init_sr_results), jnp.array(lst_K))
    )
    
    # Convert back to dictionary format
    return {
        lst_K[i]: {  # Use original tuple values
            'ret_mv_oos': final_ret_results[i],
            'X_fitted_oos': final_X_results[i],
            'sr_oos': final_sr_results[i]
        } for i in range(len(lst_K))
    }

def test_tensor_one_window():
    """Test the Tensor_One_Window function with synthetic data."""
    
    # Create synthetic data
    T_fit = 100  # Time steps for fitting
    T_oos = 20   # Out-of-sample time steps
    max_lag = 5  # Number of lags
    num_ptf = 10 # Number of portfolios
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random_seed = 42
    n_iter_max = 100
    
    # Generate random tensors
    X_fit = jnp.array(np.random.randn(T_fit, max_lag, num_ptf))
    X_oos = jnp.array(np.random.randn(T_oos, max_lag, num_ptf))
    
    # List of K values to test (now as a tuple)
    lst_K = (1, 2, 3)
    
    print("Input shapes:")
    print(f"X_fit shape: {X_fit.shape}")
    print(f"X_oos shape: {X_oos.shape}")
    print(f"Testing K values: {lst_K}")
    
    try:
        # Run the function with explicit random_seed and n_iter_max
        results = Tensor_One_Window(
            X_fit, 
            X_oos, 
            lst_K,
            random_seed=random_seed,
            n_iter_max=n_iter_max
        )
        
        # Check results
        print("\nResults:")
        for K in lst_K:
            print(f"\nK = {K}:")
            print(f"ret_mv_oos shape: {results[K]['ret_mv_oos'].shape}")
            print(f"X_fitted_oos shape: {results[K]['X_fitted_oos'].shape}")
            print(f"Sharpe ratio: {results[K]['sr_oos']:.4f}")
            
            # Basic sanity checks
            assert results[K]['ret_mv_oos'].shape == (T_oos,)
            assert results[K]['X_fitted_oos'].shape == (T_oos, max_lag, num_ptf)
            assert isinstance(results[K]['sr_oos'], float)
            
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    test_tensor_one_window()