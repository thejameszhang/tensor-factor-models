import argparse
import jax.numpy as jnp
from tfm.utils._constants import *

def load_data(data_file_name, industry = False):
    D = jnp.load(data_file_name)
    C, R = (D['C']).astype(dtype_compact), (D['R']).astype(dtype_compact)#,  D['factor_names'], D['idx_month'], D['idx_stock']
    factor_names, idx_month, idx_stock = D['factor_names'], D['idx_month'], D['idx_stock']

    # remove industries
    if industry:
        ind_chars = [f'I_{i}' for i in range(1, n_ind_default)]
        industry_indicator_idx = [factor_names.tolist().index(i) for i in ind_chars]
        char_idx = [i for i in range(len(factor_names)) if i not in industry_indicator_idx] # indexes of all characteristics and the const
        C = C[..., char_idx]
        factor_names = factor_names[char_idx]
    
    # filter firms with longest time series:
    z = jnp.any(C != 0, axis=3)[:, 0, :].sum(axis=0)
    ix = z > 300
    C = C[:, :, ix, :]
    R = R[:, :, ix]
    
    print(C.shape, R.shape)

    print(not jnp.isnan(C).any())

    # cov estimators
    # cov_est_pca_cache_file = f"{config_data['cache_path']}/cov_est_{config_data['name']}.npz"
    cov_estimator = None #PCACovEstimator(R, n_pcs=cov_pca_n_pcs_default, cache_file=cov_est_pca_cache_file)

    return {'C': C, 
            'R': R, 
            'factor_names': factor_names, 
            'idx_month': idx_month, 
            'idx_stock': idx_stock,
            'cov_estimator': cov_estimator}

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')