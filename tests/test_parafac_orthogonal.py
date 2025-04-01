import jax
import jax.numpy as jnp
import numpy as np
from tfm.parafac_orthogonal import parafac_orthogonal

def test_orthogonality_mode_0():
    """Test that PARAFAC with orthogonality constraint in mode 0 produces orthogonal factors."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create a random 3D tensor
    shape = (10, 15, 20)  # Time x Lag x Characteristics
    rank = 5
    tensor = jax.random.normal(key, shape)
    
    # Run PARAFAC with orthogonality in mode 0
    weights, factors = parafac_orthogonal(
        tensor=tensor,
        rank=rank,
        orthogonal_mode=0,
        random_state=42,
        n_iter_max=100,
        tol=1e-8
    )
    
    # Get the factor matrix from mode 0
    F = factors[0]  # This should be orthogonal
    
    # Compute correlation matrix
    corr_matrix = F.T @ F
    
    # Create identity matrix of same size
    identity = jnp.eye(rank)
    
    # Check if correlation matrix is close to identity
    diff = jnp.abs(corr_matrix - identity)
    max_diff = jnp.max(diff)
    print(f"Maximum deviation from identity matrix: {max_diff:.2e}")
    
    # Print the correlation matrix for inspection
    print("\nCorrelation matrix of mode 0 factors:")
    print(corr_matrix)
    
    # Assert that maximum difference is small
    assert max_diff < 1e-6, f"Factors not orthogonal! Max difference from identity: {max_diff:.2e}"
    
    # Also verify that the reconstruction error is reasonable
    reconstruction = jnp.zeros_like(tensor)
    for r in range(rank):
        reconstruction += weights[r] * jnp.einsum('i,j,k->ijk', 
                                                factors[0][:, r],
                                                factors[1][:, r],
                                                factors[2][:, r])
    
    rel_error = jnp.linalg.norm(tensor - reconstruction) / jnp.linalg.norm(tensor)
    print(f"\nRelative reconstruction error: {rel_error:.4f}")
    
    return weights, factors, rel_error

if __name__ == "__main__":
    weights, factors, rel_error = test_orthogonality_mode_0() 