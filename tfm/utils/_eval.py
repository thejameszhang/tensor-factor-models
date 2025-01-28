import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from typing import List

def calc_eval_metrics(X_in, X_rc):

    X_rc_1ex = X_rc[:,0,:]

    eps = X_in - X_rc # dim: (T, max_lag, num_ptf)
    eps_1ex = (X_in - X_in[:, 0, :][:,np.newaxis,:]) - (X_rc - X_rc_1ex[:,np.newaxis,:])

    ### metrics, not aggregated
    # pricing error
    alpha = eps.mean(axis=0) # dim: (max_lag, num_ptf)
    alpha_1ex = eps_1ex.mean(axis=0) # dim: (max_lag, num_ptf)

    # unexplained variance
    var_resid = (eps**2).mean(axis=0) - alpha**2 # dim: (max_lag, num_ptf)
    var_resid_1ex = (eps_1ex**2).mean(axis=0) - alpha_1ex**2 # dim: (max_lag, num_ptf)
    

    ### normalized averaged metrics
    # XS-alpha
    xs_alpha = np.sqrt((alpha**2).mean()) # scalar
    xs_alpha_1ex = np.sqrt((alpha_1ex**2).mean()) # scalar

    xs_alpha_ptf  = np.sqrt((alpha**2).mean(axis=0)) # dim: (num_ptf, )
    xs_alpha_ptf_1ex  = np.sqrt((alpha_1ex**2).mean(axis=0)) # dim: (num_ptf, )
    
    xs_alpha_lag  = np.sqrt((alpha**2).mean(axis=1)) # dim: (max_lag, )
    xs_alpha_lag_1ex  = np.sqrt((alpha_1ex**2).mean(axis=1)) # dim: (max_lag, )

    # unexplained var
    sigma_eps = np.sqrt(var_resid.mean()) \
        /np.sqrt(np.var(X_in, axis=0).mean()) # scalar
    sigma_eps_1ex = np.sqrt(var_resid_1ex.mean())\
        /np.sqrt(np.var(X_in - X_in[:,0,:][:,np.newaxis,:], axis=0).mean()) # scalar
    
    
    # fitting error decomposition into lag and cross components
    eps_cross=eps[:,0,:]
    eps_lag=eps-eps_cross[:,np.newaxis,:]

    eps_cross_rms=(eps_cross**2).mean(axis=0)**.5
    eps_lag_rms=(eps_lag**2).mean(axis=0)**.5
    
    
    # fitting rmse averaged over two dimensions
    dict_rmse={}
    lst_mean_axis=[(1,2),(0,2),(0,1)]
    lst_rmse_dim_name=['rmse_time', 'rmse_lag', 'rmse_portfolio']

    for mean_axis, dim_name in zip(lst_mean_axis,lst_rmse_dim_name):
        rmse_dim=np.sqrt(((X_in-X_rc)**2).mean(axis=mean_axis))
        dict_rmse[dim_name]=rmse_dim
    
    ### record results
    dict_eval = {
                 'alpha':alpha,
                'alpha_1ex':alpha_1ex,
                'var_resid':var_resid,
                'var_resid_1ex':var_resid_1ex,
                 
                'xs_alpha':xs_alpha,
                'xs_alpha_1ex':xs_alpha_1ex,
                'xs_alpha_ptf':xs_alpha_ptf,
                'xs_alpha_ptf_1ex':xs_alpha_ptf_1ex,
                 'xs_alpha_lag':xs_alpha_lag,
                 'xs_alpha_lag_1ex':xs_alpha_lag_1ex,
                 
                'sigma_eps':sigma_eps,
                'sigma_eps_1ex':sigma_eps_1ex,
                 'eps_cross_rms':eps_cross_rms,
                 'eps_lag_rms':eps_lag_rms
                }
    dict_eval.update(dict_rmse)
    
    return dict_eval

def print_matrix(A,row_title=None,col_title=None,print_multiplier=1, fmt='{:.3f}'):
    '''
    print(print_matrix(A))
    '''
    str_out=''
    (num_row,num_col)=A.shape
    
    if row_title is not None:
        assert len(row_title)==num_row
        
    if col_title is not None:
        for i,title in enumerate(col_title):
            if i==len(col_title)-1:
                str_out+=title+'\\\\'
            else:
                str_out+=title+'&'
        str_out+='\n'
        
    for i in range(0,num_row):
        str_temp=''
        if row_title is not None:
            str_temp+=row_title[i]+'&'
        for j in range(0,num_col):
            str_temp+=fmt.format(A[i,j]*print_multiplier)
            if j==num_col-1:
                str_temp+='\\\\'
            else:
                str_temp+='&'
        #str_temp=str_temp.replace('<','$<$').replace('>','$>$')
        str_out+=str_temp+'\n'
    return str_out

def get_plot_ylim(dict_eval_method, lst_K, key, q_lb=0.1):
    q_ub=1-q_lb
    
    ylim = [float('inf'), -float('inf')]
    for K in lst_K:
        ylim[0] = min(ylim[0], np.quantile(dict_eval_method[K]['eval'][key], q_lb))
        ylim[1] = max(ylim[1], np.quantile(dict_eval_method[K]['eval'][key], q_ub))
        
    return ylim

    
def collect_eval_metric(dict_eval_method, lst_K, key):
    temp = []
    for K in lst_K:
        temp.append(np.array(dict_eval_method[K]['eval'][key]))
    return np.stack(temp, axis=-1)

def plot2x2(sr: jnp.ndarray, 
            lst_K: List[int], 
            max_horizon: int, 
            method: str, 
            window_size: int, 
            lag: int,
            x: int, 
            y: int, 
            fig, 
            axes,
            start_date: str,
            end_date: str,
            xlabel: str = 'Horizon (Months)',
            ylabel: str = 'Annualized Sharpe Ratio'):
    """
    Plots in a 2x2 grid for Multi-horizon experiments. Dates are intepreted as days where investment decisions happen.
    t + window_size for all t in dataset periods - window_size - max_horizon + 1
    """
    axes[x][y].set_title(f'{method}: Window size {window_size}, Lag {lag}')
    axes[x][y].text(0.5, 0.95, f'{start_date} to {end_date}', transform=axes[x][y].transAxes, ha='center', fontsize=10)
    for i in range(len(lst_K)):
        temp = sr[:, i] * jnp.sqrt(1 / jnp.arange(1, max_horizon + 1)) * jnp.sqrt(12)
        axes[x][y].plot(temp, linestyle='-', label=f'K={lst_K[i]}')
    axes[x][y].set_xticks(jnp.arange(0, max_horizon, 5), labels=jnp.arange(1, max_horizon + 1, 5))
    axes[x][y].set_xlabel(xlabel)
    axes[x][y].set_ylabel(ylabel)
    axes[x][y].legend()
    axes[x][y].grid(True)

def pca_plot2x2(sr: jnp.ndarray, 
            lst_K: List[int], 
            max_horizon: int, 
            method: str, 
            window_size: int, 
            lag: int,
            x: int, 
            y: int, 
            fig, 
            axes,
            start_date: str = "",
            end_date: str = "",
            xlabel: str = 'Horizon (Months)',
            ylabel: str = 'Annualized Sharpe Ratio'):
    """
    Plots in a 2x2 grid for Multi-horizon experiments. Dates are intepreted as days where investment decisions happen.
    t + window_size for all t in dataset periods - window_size - max_horizon + 1
    """
    title = f'{method}: Window size {window_size}' if not lag else f'{method}: Window size {window_size}, Lag {lag}'
    axes[x][y].set_title(title)
    if start_date and end_date:
        axes[x][y].text(0.5, 0.95, f'{start_date} to {end_date}', transform=axes[x][y].transAxes, ha='center', fontsize=10)
    for i in range(len(lst_K)):
        temp = sr[:, i] * jnp.sqrt(1 / jnp.arange(1, max_horizon + 1)) * jnp.sqrt(12)
        axes[x][y].plot(temp, linestyle='-', label=lst_K[i] if i == len(lst_K)-1 else f'K={lst_K[i]}')
    axes[x][y].set_xticks(jnp.arange(0, max_horizon, 5), labels=jnp.arange(1, max_horizon + 1, 5))
    axes[x][y].set_xlabel(xlabel)
    axes[x][y].set_ylabel(ylabel)
    axes[x][y].legend()
    axes[x][y].grid(True)

def plot(sr: jnp.ndarray, 
            lst_K: List[int], 
            max_horizon: int, 
            method: str, 
            window_size: int, 
            lag: int,
            fig, 
            axes,
            start_date: str,
            end_date: str,
            xlabel: str = 'Horizon (Months)',
            ylabel: str = 'Annualized Sharpe Ratio',
            gamma: int = 0):
    """
    """
    if lag == 0:
        axes.set_title(f'{method}: Window size {window_size}, Gamma={gamma}')
    else:
        if 'PCA' in method:
            axes.set_title(f'{method}: Window size {window_size}, Lag {lag}, Gamma={gamma}')
        else:
            axes.set_title(f'{method}: Window size {window_size}, Lag {lag}')
    axes.text(0.5, 0.95, f'{start_date} to {end_date}', transform=axes.transAxes, ha='center', fontsize=10)
    for i in range(len(lst_K)):
        temp = sr[:, i] * jnp.sqrt(1 / jnp.arange(1, max_horizon + 1)) * jnp.sqrt(12)
        axes.plot(temp, linestyle='-', label=f'K={lst_K[i]}')
    axes.set_xticks(jnp.arange(0, max_horizon, 5), labels=jnp.arange(1, max_horizon + 1, 5))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend()
    axes.grid(True)