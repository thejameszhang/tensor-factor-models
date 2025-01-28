import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os
from tqdm.auto import tqdm
import argparse
import pprint
import jax
import jax.numpy as jnp
from tfm.utils._pca import RPPCA, SDF_construct

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='char_anom', help='char_anom, scs, wrds' )
parser.add_argument('--lst_gamma', type=str, default='-1', help='gamma parameter' )
args = parser.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

config = args.dataset
lst_gamma = [int(x) for x in args.lst_gamma.split(',')]

assert args.dataset in ('char_anom', 'wrds', 'scs')
os.environ['CONFIG'] = config

from tfm.utils._load_configs import *

# Read tensor data
X = jnp.load(f'{dir_input}/mat_ptf_re_lag_{max_lag}.npz')['mat_ptf_re_rank']
params = jnp.load(f'{dir_input}/dict_param_lag_{max_lag}.pkl', allow_pickle=True)
num_ptf = X.shape[2]

if config == 'char_anom':
    bin_labels, _, _, max_lag, frac_longshort, all_dates, start_date_maxlag = params.values()
    dates_fit = all_dates[all_dates < start_date_oos]
elif config == 'wrds' or config == 'scs':
    bin_labels, all_dates = params['lst_char'], params['all_dates']
    dates_fit = all_dates[all_dates < start_date_oos][max_lag - 1:]

dir_out = f'../results_oos/pca_loadings/{config}/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

K_max = 9
K_plot = 9
num_fig_x = 3
num_fig_y = int(np.ceil(K_plot / num_fig_x))
X_flatten = X.reshape(X.shape[0], -1)
figsize = (15, 5 * num_fig_y) if config in ('scs', 'char_anom') else (30, 10 * num_fig_y)

for gamma in lst_gamma:
    
    F_PCA,L_PCA = RPPCA(X=X_flatten, K=K_max, gamma=gamma)
    SDF_PCA, SDFweight_PCA = SDF_construct(F_PCA, L_PCA)
    #SR_PCA = SDF_PCA.mean(axis=0) / SDF_PCA.std(axis=0) * np.sqrt(12)

    F_PCA = np.real(F_PCA)
    L_PCA = np.real(L_PCA)
    L_3d = L_PCA.reshape(max_lag, num_ptf, K_max)


    fig=plt.figure(figsize=figsize)
    for i in range(K_plot):
        ax=fig.add_subplot(num_fig_y,num_fig_x,i+1)
        sns.heatmap(L_3d[:,:,i].T, ax=ax,
                    cmap=sns.color_palette("RdBu_r", n_colors=100),
                   vmin=-0.1, vmax=0.1);
        ax.set_yticks(np.arange(num_ptf));
        ax.set_yticklabels(bin_labels,rotation=0, fontsize=7);
        ax.set_title('Loading {}'.format(i+1));
        
    fig.savefig(dir_out+'loading_{}_rppca_gamma_{}.png'.format(input_type, gamma),
                bbox_inches='tight')