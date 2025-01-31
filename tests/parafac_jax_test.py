import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from functools import partial
from typing import List, Dict, Tuple, Optional
import tensorly as tl
from tfm.parafac_jax import *
from orig.tl_parafac_fix_intercept import parafac_fix_intercept


def compare_implementations(
    tensor_shape=(10, 10, 10),
    rank=3,
    random_state=0,
    fix_intercept_mode=0,
    overweight_mode=1,
    gamma=0.1
):
    """Compare original and JAX implementations."""
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Generate same random tensor for both implementations
    tensor_np = np.random.rand(*tensor_shape)
    tensor_jax = jnp.array(tensor_np)
    
    # Run original implementation
    cp_tensor_original = parafac_fix_intercept(
        tensor_np,
        rank=rank,
        fix_intercept_mode=fix_intercept_mode,
        overweight_mode=overweight_mode,
        gamma=gamma,
        normalize_factors=True
    )
    weights_original, factors_original = cp_tensor_original.weights, cp_tensor_original.factors
    
    weights_jax_before, factors_jax_before = initialize_factors_svd(tensor_jax, rank)

    # Run JAX implementation
    weights_jax, factors_jax = parafac_enhanced(
        tensor_jax,
        rank=rank,
        random_state=random_state,
        fix_intercept_mode=fix_intercept_mode,
        overweight_mode=overweight_mode,
        gamma=gamma,
        normalize_factors=True,
        nonnegative=False
    )
    
    # Convert JAX arrays to numpy for comparison
    weights_jax = np.array(weights_jax)
    factors_jax = [np.array(f) for f in factors_jax]
    
    # Reconstruct tensors
    reconstructed_original = tl.cp_to_tensor((weights_original, factors_original))
    reconstructed_jax = np.array(reconstruct_tensor(jnp.array(weights_jax), [jnp.array(f) for f in factors_jax]))
    
    # Calculate errors
    error_original = np.linalg.norm(tensor_np - reconstructed_original) / np.linalg.norm(tensor_np)
    error_jax = np.linalg.norm(tensor_np - reconstructed_jax) / np.linalg.norm(tensor_np)
    
    # Compare factors (after aligning signs)
    factors_diff = []
    for f_orig, f_jax in zip(factors_original, factors_jax):
        # Align signs
        signs = np.sign(np.sum(f_orig * f_jax, axis=0))
        f_jax_aligned = f_jax * signs[None, :]
        factors_diff.append(np.linalg.norm(f_orig - f_jax_aligned) / np.linalg.norm(f_orig))
    
    print(f"Comparison Results:")
    print(f"Original implementation error: {error_original:.6f}")
    print(f"JAX implementation error:     {error_jax:.6f}")
    print(f"Relative difference in errors: {abs(error_original - error_jax) / error_original:.6f}")
    print("\nFactor differences (relative norm):")
    for i, diff in enumerate(factors_diff):
        print(f"Mode {i}: {diff:.6f}")
    
    # Check if first column is ones where required
    if fix_intercept_mode >= 0:
        print(f"\nChecking fixed intercept mode {fix_intercept_mode}:")
        print("Original first column:", factors_original[fix_intercept_mode][:5, 0])
        print("JAX first column:     ", factors_jax[fix_intercept_mode][:5, 0])
        print(factors_jax[fix_intercept_mode])
    
    return {
        'error_original': error_original,
        'error_jax': error_jax,
        'factors_diff': factors_diff,
        'weights_original': weights_original,
        'weights_jax': weights_jax,
        'factors_original': factors_original,
        'factors_jax': factors_jax
    }
    
    

if __name__ == "__main__":
    # Set global random seed
    RANDOM_SEED = 42
    
    print("Test 1: Basic decomposition")
    results1 = compare_implementations(
        tensor_shape=(10, 10, 10),
        rank=3,
        random_state=RANDOM_SEED,
        fix_intercept_mode=-1,  # No fixed intercept
        overweight_mode=-1,     # No overweighting
        gamma=0.0
    )
    
    print("\nTest 2: With fixed intercept")
    results2 = compare_implementations(
        tensor_shape=(10, 10, 10),
        rank=3,
        random_state=RANDOM_SEED,
        fix_intercept_mode=0,   # Fix first mode
        overweight_mode=-1,     # No overweighting
        gamma=0.0
    )
    
    print("\nTest 3: With overweighting")
    results3 = compare_implementations(
        tensor_shape=(10, 10, 10),
        rank=3,
        random_state=RANDOM_SEED,
        fix_intercept_mode=-1,  # No fixed intercept
        overweight_mode=1,      # Overweight second mode
        gamma=0.1
    )
    
    print("\nTest 4: With both fixed intercept and overweighting")
    results4 = compare_implementations(
        tensor_shape=(10, 10, 10),
        rank=3,
        random_state=RANDOM_SEED,
        fix_intercept_mode=0,   # Fix first mode
        overweight_mode=1,      # Overweight second mode
        gamma=0.1
    )