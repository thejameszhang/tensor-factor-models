# Overleaf Document for Results

https://www.overleaf.com/8589826629ycndpfwtmqny#9b6ace

# Tweaks from the original code

1. Setting the Jax backend using Tensorly is outdated for newer versions of Jax - it tries to import jax.config which is unallowed, so I edited the following file 
`~/conda/envs/jax/lib/python3.12/site-packages/tensorly/backend/jax_backend.py` and modified a line. 

5. Tweaked some code in `tl_src.py` to use `tl.tenalg.svd.truncated_svd` or `randomized_svd` function, as `numpy.svd` is deprecated, not sure if this could cause the difference in the heatmaps?

6. Numpy backend for tensorly seems faster than Jax backend?

7. Using `jnp.linalg.eigh` instead of the normal `jnp.linalg.eig` much faster for eigenvalue decomposition of covariance matrix


# Nov 15 Meeting

- slides 14-16 basic results $\implies$ Q: they look slightly different, check if this is okay 
- next notebook need to implement is `multiperiod_oos_one_fit_merged` which generates slides 17-18
- continue refactoring code into `tfm/`, using config files more, then writing scripts to generate all results
    - probably move config into `tfm/` soon

work with the two new datasets

resume and cv and schools over the weekend 


# Helpful reading

1. https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables to set env variables automatically in Conda environments.

2. https://www.cs.cmu.edu/afs/cs/Web/People/pmuthuku/mlsp_page/lectures/Parafac.pdf - parafac paper

3. https://tensorly.org/stable/user_guide/tensor_decomposition.html - parafac docs


when it comes to cross sectional ranking, it turns out that the relative standing is more important in explaining returns and covariances than the actual numerical values of the characteristics
