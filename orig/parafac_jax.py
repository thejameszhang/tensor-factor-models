import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple, List, Optional
from functools import partial
import numpy as np
import tensorly as tl
from orig.tl_src import initialize_cp, validate_cp_rank
from orig.tl_parafac_fix_intercept import parafac_fix_intercept

@partial(jax.jit, static_argnums=(1, 2))
def initialize_factors_svd(tensor: jnp.ndarray, rank: int, random_state: int = 0) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Initialize factor matrices using SVD."""
    weights = jnp.ones(rank) # Initial weights are different then src, but doesn't change anything
    factors = []
    
    for mode in range(tensor.ndim):
        u, s, _ = jnp.linalg.svd(matricize(tensor, mode), full_matrices=False)
        factors.append(u[:, :rank])
    
    return weights, factors

@partial(jax.jit, static_argnums=(1, 2))
def initialize_factors_random(tensor: jnp.ndarray, rank: int, random_state: int) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Initialize factor matrices using random initialization."""
    weights = jnp.ones(rank)
    shapes = [(tensor.shape[i], rank) for i in range(tensor.ndim)]
    key = jax.random.PRNGKey(random_state)
    keys = jax.random.split(key, len(shapes))
    factors = [jax.random.uniform(k, shape, minval=0.0, maxval=1.0) 
              for k, shape in zip(keys, shapes)]
    
    # Normalize initial factors
    norms = [jnp.linalg.norm(f, axis=0) for f in factors]
    weights = weights * jnp.prod(jnp.stack(norms), axis=0)
    factors = [f / (norm[None, :] + 1e-12) for f, norm in zip(factors, norms)]
    
    return weights, factors

@jax.jit
def calculate_reconstruction_error(tensor: jnp.ndarray, weights: jnp.ndarray, factors: List[jnp.ndarray]) -> jnp.ndarray:
    """Calculate reconstruction error."""
    reconstructed = reconstruct_tensor(weights, factors)
    diff = tensor - reconstructed
    return jnp.sqrt(jnp.sum(diff ** 2)) / jnp.sqrt(jnp.sum(tensor ** 2))

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def parafac_enhanced(
    tensor: jnp.ndarray,
    rank: int,
    random_state: int = 0,
    fixed_modes: Optional[List[int]] = None,
    fix_intercept_mode: int = -1,
    overweight_mode: int = -1,
    gamma: float = 0.0,
    normalize_factors: bool = False,
    n_iter_max: int = 100,
    tol: float = 1e-8,
    init: str = 'svd'
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Enhanced CP decomposition with improved stability."""
    
    # Initialize factors
    if init == 'svd':
        weights, factors = initialize_factors_svd(tensor, rank, random_state)
        # Weights all 1s, different to src, but changing src doesn't affect final src res
        # Factors initialized to the same
    else:
        weights, factors = initialize_factors_random(tensor, rank, random_state)
    
    # Handle fixed modes
    modes_list = list(range(tensor.ndim))
    if fixed_modes is not None:
        modes_list = [m for m in modes_list if m not in fixed_modes]
    
    # Apply overweighting if specified
    if overweight_mode != -1:
        mean_tensor = jnp.mean(tensor, axis=overweight_mode, keepdims=True)
        tensor_work = tensor + gamma * mean_tensor
    else:
        tensor_work = tensor

    # Checkpoint: tensors, factors, weights, modes_list same up to here
    # Body fun executed n_iter_max times
    def body_fun(carry, _):
        factors, weights, prev_error = carry
        
        # Store previous reconstruction for convergence check
        prev_reconstruction = reconstruct_tensor(weights, factors)
        
        # Update each factor matrix
        for mode in modes_list:
            tensor_use = tensor_work if mode != overweight_mode else tensor
            
            if mode == fix_intercept_mode:
                new_factor = update_factor_fixed_intercept(tensor_use, factors, mode, weights)
            else:
                new_factor = update_factor(tensor_use, factors, mode, weights)
            
            factors = [
                new_factor if i == mode else f
                for i, f in enumerate(factors)
            ]
        
        # Normalize after all modes are updated
        if normalize_factors:
            # Calculate norms of each factor
            norms = [jnp.linalg.norm(f, axis=0) for f in factors]

            # Update weights with product of norms
            weights = weights * jnp.prod(jnp.stack(norms), axis=0)
            
            # Normalize factors, avoiding division by zero
            factors = [
                f / (norm[None, :] + 1e-12)  # Changed from maximum to addition for stability
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

@partial(jax.jit, static_argnums=(2,))
def update_factor_fixed_intercept(tensor: jnp.ndarray, factors: List[jnp.ndarray], mode: int, weights: jnp.ndarray) -> jnp.ndarray:
    """Update factor matrix while keeping first column fixed to ones."""
    # Get regular update
    updated_factor = update_factor(tensor, factors, mode, weights)
    
    # Fix first column to ones (not normalized)
    updated_factor = updated_factor.at[:, 0].set(1.0)
    
    # Let normalization happen later if needed
    return updated_factor

@partial(jax.jit, static_argnums=(2,))
def update_factor(tensor: jnp.ndarray, factors: List[jnp.ndarray], mode: int, weights: jnp.ndarray) -> jnp.ndarray:
    """Update factor matrix using ALS with improved numerical stability."""
    other_factors = [f for i, f in enumerate(factors) if i != mode]
    # kr_product = khatri_rao_product(other_factors[::-1])
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

@jax.jit
def reconstruct_tensor(weights: jnp.ndarray, factors: List[jnp.ndarray]) -> jnp.ndarray:
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

@partial(jax.jit, static_argnums=(1,))
def matricize(tensor: jnp.ndarray, mode: int) -> jnp.ndarray:
    """Matricize the tensor along the specified mode."""
    return jnp.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

@jax.jit
def khatri_rao_product(matrices: List[jnp.ndarray]) -> jnp.ndarray:
    """Compute the Khatri-Rao product of a list of matrices."""
    n_cols = matrices[0].shape[1]
    result = matrices[0]
    for matrix in matrices[1:]:
        result = jnp.einsum('ir,jr->ijr', result, matrix).reshape(-1, n_cols)
    return result

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
    
    # Run JAX implementation
    weights_jax, factors_jax = parafac_enhanced(
        tensor_jax,
        rank=rank,
        random_state=random_state,
        fix_intercept_mode=fix_intercept_mode,
        overweight_mode=overweight_mode,
        gamma=gamma,
        normalize_factors=True
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
    # if fix_intercept_mode >= 0:
    #     print(f"\nChecking fixed intercept mode {fix_intercept_mode}:")
    #     print("Original first column:", factors_original[fix_intercept_mode][:5, 0])
    #     print("JAX first column:     ", factors_jax[fix_intercept_mode][:5, 0])
    
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