"""
Example usage: `python3 oos_one_fit_ver_1.py --dataset scs`
============================================================

Author: James Zhang
Date: November 2024
"""
#%%
import argparse
import os
import pprint
import jax
from tfm.utils._data import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data setting
parser.add_argument('--dataset', type=str, default='scs', help='char_anom, scs, wrds' )

# oos settings
parser.add_argument('--lst_K', type=str, default='1,3,5,10,20,30,40',
    help='number of factors separated by ","' )

args = parser.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

# from types import SimpleNamespace
# args = SimpleNamespace(dataset='scs', lst_K='1,3,5,10,20,30')
config = args.dataset
os.environ['CONFIG'] = config
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
jax.config.update('jax_platform_name', 'gpu')
print("JAX is using device:", jax.devices()[0])
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
# import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from tfm.utils._load_configs import *
from tfm.models.pca import RPPCA_One_Window_Jax, RPPCA_One_Window
from tfm.models.tensor import Tensor_One_Window_One_K
from tfm.utils._eval import *

#%%
# ### load data

# Read tensor data
X = jnp.load(f'{dir_input}/mat_ptf_re_lag_{max_lag}.npz')['mat_ptf_re_rank']
# Read the dictionary parameters
params = jnp.load(f'{dir_input}/dict_param_lag_{max_lag}.pkl', allow_pickle=True)
assert params['max_lag'] == max_lag

print(f"MAX LAG: {max_lag} months")

if config == 'char_anom':
    bin_labels, _, _, max_lag, frac_longshort, all_dates, start_date_maxlag = params.values()
    # Read FF market factor
    df_ff = jnp.load(f'{dir_input}/df_FF_factors.pkl', allow_pickle=True) * 0.01 
    df_ff = df_ff.loc[all_dates]
    # Get excess returns ie. excess to normal factor
    if input_type == 'rx':
        print("Excess")
        X = X - X[:, 0, :][:, jnp.newaxis, :]
    # if normalize_ret:
    #     print("Normalizing")
    #     X = X / X.std(axis=0, keepdims=True) * df_ff['Mkt-RF'].std()
    dates_fit = all_dates[all_dates < start_date_oos]
elif config == 'wrds' or config == 'scs':
    bin_labels, all_dates = params['lst_char'], params['all_dates']
    dates_fit = all_dates[all_dates < start_date_oos][max_lag - 1:]
    
T = len(all_dates)
start_date = all_dates[0]
dates_oos = all_dates[all_dates >= start_date_oos]
T_fit = len(dates_fit)
T_oos = len(dates_oos)


# Find indexes of intercept terms to remove
intercept_terms = ['I_1', 'I_2', 'I_3', 'I_4', 'I_5', 'I_6', 'I_7', 'I_8', 'I_9', 'I_10', 'I_11', 'const']
idx_remove = [i for i, label in enumerate(bin_labels) if label in intercept_terms]

if idx_remove:
    # Remove from bin_labels
    bin_labels = [label for i, label in enumerate(bin_labels) if i not in idx_remove]
    
    # Get indices to keep (all indices except those in idx_remove)
    idx_keep = jnp.array([i for i in range(X.shape[2]) if i not in idx_remove])
    
    # Use jnp.take to select columns to keep from X
    X = jnp.take(X, idx_keep, axis=2)

assert not jnp.isnan(X).any()

num_ptf = X.shape[2]

#%%
print(f'Shape of X: {X.shape}, (Months, Lag, Factor)')
X_fit, X_oos = X[:T_fit], X[T_fit:]
print(f'In-Sample Start = {dates_fit[0]}, Out-Of-Sample Start = {dates_oos[0]}')

# Output directory
dir_out = f'../results_oos/one_fit/{config}/fig_onefit_oos_{input_type}_{spec}_ver{idx_ver}/{max_lag}/'
dir_out_table = f'../results_oos/one_fit/{config}/tbl_onefit_oos_{input_type}_{spec}_ver{idx_ver}/{max_lag}/'

if not os.path.exists(dir_out):
    os.makedirs(dir_out)
if not os.path.exists(dir_out_table):
    os.makedirs(dir_out_table)

#%%
### plot mean return and fitted mean return
X_mean = X[T_fit:].mean(axis=0)
q_ylim = 0.01
ylim_mean = [np.quantile(X_mean, q_ylim), np.quantile(X_mean, 1 - q_ylim)]
# mean oos return
fig = plt.figure(figsize=(13,13))
ax = fig.add_subplot(1,1,1)    
sns.heatmap(
    X_mean.T, annot=False, fmt='.0e', ax=ax, cmap=sns.color_palette("RdBu_r", n_colors=100),
    vmin=ylim_mean[0], vmax=ylim_mean[1]
);
ax.set_yticks(np.arange(len(bin_labels)))
ax.set_yticklabels(bin_labels, rotation=0);
ax.set_xticks(np.arange(0, max_lag, 12))
ax.set_xticklabels([str(i) for i in np.arange(1,max_lag+1,12)],rotation=0)
ax.set_xlabel('Lags');
filename='term_structure_mean_ret.pdf'
fig.savefig(dir_out+filename, bbox_inches='tight')


# In[6]:


# OOS settings
lst_K = [int(x) for x in args.lst_K.split(',')]
if config == 'scs':
    lst_K = [x for x in lst_K if x <= X.shape[2]]

print("Processing K values: ", lst_K)

##############
## PCA
##############
dict_fit_oos = defaultdict(dict)

X_flatten = X[:, :max_lag, :].reshape(X.shape[0], -1)
X_onelag_pca = X[:, 1, :] if fit_rx else X[:, 0, :] # False in this nb

print("PCA One Lag")

# Advantage of doing it this way is that we only have to compute eigendecomposition once. 
dict_fit_pca_onelag = RPPCA_One_Window_Jax(
    X_onelag_pca[:T_fit], X_onelag_pca[T_fit:], lst_K, gamma=-1
)

print("PCA")

dict_fit_pca = RPPCA_One_Window_Jax(
    X_flatten[:T_fit], X_flatten[T_fit:], lst_K, gamma=-1
)


for idx_K, K in enumerate(lst_K):
    dict_fit_oos['PCA One Lag'][K] = dict_fit_pca_onelag[K]
    dict_fit_pca[K]['X_fitted_oos'] = dict_fit_pca[K]['X_fitted_oos'].reshape(T_oos, max_lag, num_ptf)
    dict_fit_oos['PCA'][K] = dict_fit_pca[K]

#%%
print("Tensor")
pbar = tqdm(total=len(lst_K))
for K in lst_K:
    dict_fit_oos['Tensor'][K] = Tensor_One_Window_One_K(X[:T_fit], X[T_fit:], K, random_seed=random_seed)
    pbar.update(1)

# mean fitted oos return
for method in ['Tensor', 'PCA']:
    for idx_K, K in enumerate(lst_K):
        # get fitted expected return
        X_fit_oos=dict_fit_oos[method][K]['X_fitted_oos']
        expected_re_fit=X_fit_oos.mean(axis=0)

        # plot
        fig=plt.figure(figsize=(13,13))
        ax=fig.add_subplot(1,1,1)    
        sns.heatmap(expected_re_fit.T,
                    annot=False,
                    fmt='.0e',
                    ax=ax,
                    cmap=sns.color_palette("RdBu_r", n_colors=100),
                    vmin=ylim_mean[0], vmax=ylim_mean[1]
        );
        ax.set_yticks(np.arange(len(bin_labels)))
        ax.set_yticklabels(bin_labels, rotation=0);
        ax.set_xticks(np.arange(0,max_lag,12))
        ax.set_xticklabels([str(i) for i in np.arange(1,max_lag+1,12)],rotation=0)
        ax.set_xlabel('Lags');

        filename='term_structure_{}_fitted_ret_K{}.pdf'\
            .format(method, K)
        #print(filename)
        fig.savefig(dir_out+filename, bbox_inches='tight')
        plt.close('all')


for method in ['Tensor', 'PCA']: 
    for idx_K, K in enumerate(lst_K):

        X_rc=dict_fit_oos[method][K]['X_fitted_oos']

        dict_eval_temp = calc_eval_metrics(X[T_fit:], X_rc)
        dict_fit_oos[method][K]['eval'] = dict_eval_temp


# In[11]:

print("Plotting Alphas")
### alpha
print_multiplier=1

for method in ['Tensor','PCA']:
    for K in lst_K:
        dict_eval=dict_fit_oos[method][K]['eval']
        latex = print_matrix(dict_eval['alpha'], fmt='{:.2e}', 
                        col_title=[str(s) for s in bin_labels],
                        row_title=['Lag {}'.format(i+1) for i in range(max_lag)],
                         print_multiplier=print_multiplier)
        
        text_file = open(dir_out_table+'tbl_{}_alpha_K_{}.txt'.format(method, K), 'w')
        text_file.write('Multiplier: {:.2e}; Method: {}; K: {}\n\n'.format(print_multiplier, method, K))
        text_file.write(latex)
        text_file.close()


# In[12]:


### table for xs-alpha, averaged over lag or ptf dim

print("Plotting Excess Alphas")

print_multiplier=1e3

for method in ['Tensor', 'PCA']:
    for postfix in ['','_1ex']:
        dict_eval=dict_fit_oos[method]
        
        latex_ptf=print_matrix(collect_eval_metric(dict_eval, lst_K, 'xs_alpha_ptf'+postfix),
                     row_title=[str(s) for s in bin_labels],
                     col_title=[str(K) for K in lst_K],
                     print_multiplier=print_multiplier,
                     fmt='{:.2f}')


        latex_lag=print_matrix(collect_eval_metric(dict_eval, lst_K, 'xs_alpha_lag'+postfix),
                     row_title=[str(L) for L in range(1,1+max_lag)],
                     col_title=[str(K) for K in lst_K],
                     print_multiplier=print_multiplier,
                     fmt='{:.2f}')

        # save rms alpha as table in txt for latex
        text_file = open(dir_out_table+'tbl_{}_xs_alpha_ptf{}.txt'.format(method, postfix), 'w')
        text_file.write('Multiplier: {:.2e}; Method: {}; \n\n'.format(print_multiplier, method))
        text_file.write(latex_ptf)
        text_file.close()

        text_file = open(dir_out_table+'tbl_{}_xs_alpha_lag{}.txt'.format(method, postfix), 'w')
        text_file.write('Multiplier: {:.2e}; Method: {}; \n\n'.format(print_multiplier, method))
        text_file.write(latex_lag)
        text_file.close()


# In[13]:

print("Plotting Unexplained Variances")
### sigma_eps unexplained variance
print_multiplier = 1

for postfix in ['','_1ex']:
    

    metric_tensor = collect_eval_metric(dict_fit_oos['Tensor'],
                                        lst_K, 'sigma_eps'+postfix)
    metric_pca = collect_eval_metric(dict_fit_oos['Tensor'],
                                     lst_K, 'sigma_eps'+postfix)
    metric_pca = np.real(metric_pca)
    metric = np.stack((metric_tensor,metric_pca), axis=-1)

    latex = print_matrix(metric,
                 col_title=['Tensor', 'PCA'],
                 row_title=[str(K) for K in lst_K],
                 print_multiplier=print_multiplier,
                 fmt='{:.2f}')

    text_file = open(dir_out_table+'tbl_{}_sigma_eps{}.txt'.format(method, postfix), 'w')
    text_file.write('Multiplier: {:.2e}\n\n'.format(print_multiplier))
    text_file.write(latex)
    text_file.close()


# In[14]:


### table for eps_cross_rms
print_multiplier=1e2

for method in ['Tensor','PCA']:
    latex=print_matrix(collect_eval_metric(dict_fit_oos[method],
                                           lst_K, 'eps_cross_rms'),
                                         fmt='{:.2f}', 
                                        col_title=[str(K) for K in lst_K],
                                         row_title=[str(s) for s in bin_labels],
                                         print_multiplier=print_multiplier)
    
    text_file = open(dir_out_table+'tbl_{}_rms_eps_cross.txt'.format(method), 'w')
    text_file.write('Multiplier: {:.2e}; Method: {} \n\n'.format(print_multiplier, method))
    text_file.write(latex)
    text_file.close()


# In[15]:

print("Plotting Sharpe Ratios")
#ylim=[-0.1, 1.9]
filename = 'sr.pdf'

dict_fit_oos['Tensor'][1]['sr_oos'] = dict_fit_oos['Tensor'][1]['sr_oos'][0]

fig=plt.figure(figsize=(7,6))
ax=fig.add_subplot(1,1,1)

for method in ['Tensor', 'PCA', 'PCA One Lag']:
    ax.plot(lst_K, np.array([dict_fit_oos[method][K]['sr_oos'] * np.sqrt(12) for K in lst_K]),
            '--o',label=method)
ax.legend(fontsize=13);
ax.set_ylabel('SR')
ax.set_xlabel('Number of Factors')
ax.set_xticks(lst_K);
#ax.set_ylim(ylim)

fig.savefig(f'{dir_out}{filename}', bbox_inches='tight')


# In[16]:


### xs-alpha and sigma_eps

for key in ['xs_alpha','sigma_eps']:

    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(1,1,1)
    for method in ['Tensor','PCA']:
        metric=np.real(collect_eval_metric(dict_fit_oos[method], lst_K, key))
        ax.plot(lst_K, metric,'--o',
                label=method)
    ax.set_xlabel('Number of factors');
    ax.legend();
    ax.set_xticks(lst_K);
    fig.savefig(dir_out+'{}.pdf'.format(key), bbox_inches='tight')


# In[17]:


### plot term structure of alpha
lst_key = ['alpha','var_resid', 'eps_lag_rms']
filename = 'term_structure_{}_{}_K{}.pdf'
method='Tensor'


for method in ['Tensor', 'PCA']: 
    for key in lst_key:
        ylim = get_plot_ylim(dict_fit_oos[method], lst_K, key, q_lb=0.1)

        for K in lst_K:
            fig=plt.figure(figsize=(13,13))
            ax=fig.add_subplot(1,1,1)    
            sns.heatmap(dict_fit_oos[method][K]['eval'][key].T,
                        annot=False,
                        fmt='.0e',
                        ax=ax,
                        cmap=sns.color_palette("RdBu_r", n_colors=100),
                       vmin=ylim[0], vmax=ylim[1]);

            ax.set_yticks(np.arange(len(bin_labels)))
            ax.set_yticklabels(bin_labels ,rotation=0);

            ax.set_xticks(np.arange(0,max_lag,12))
            ax.set_xticklabels([str(i) for i in np.arange(1,max_lag+1,12)],rotation=0)
            ax.set_xlabel('Lags');

            fig.savefig(dir_out+filename.format(method, key, K), bbox_inches='tight')
            plt.close('all')



# %%
