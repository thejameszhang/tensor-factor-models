import jax.numpy as jnp
import matplotlib.pyplot as plt
from tfm.utils._eval import plot
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)

# settings
datasets = ['char_anom', 'scs', 'wrds', 'ff']
models = ["tensor", "pca"]

dataset_to_start_year = {'char_anom': 1992, 'scs': 1982, 'ff': 1982, 'wrds': 1985}
start_date_oos = pd.to_datetime('2005-01-01')
max_lag = 120
max_horizon = 36
window_size = 120
params_in = '/home/james/projects/tsfc/code/code_11092024/organized_data/organized_data'
gamma = 0

LAG = 36

for post2005 in [True, False]:
    for dataset in datasets:
        lst_K = [1, 2, 3, 4, 5] if dataset == 'ff' else [1, 3, 5, 10, 15, 20, 25]
        print("Post 2005: ", post2005, dataset)
        # more settings
        if post2005:
            start_year = 2005
            sample = 'post-2005'
        else:
            start_year = dataset_to_start_year[dataset]
            sample = 'full'

        # File read paths
        dir_in = f'/home/james/projects/tsfc/code/code_11092024/results_oos/multiperiod/{dataset}/'
        dir_out = f'/home/james/projects/tsfc/code/code_11092024/notes/figures/{sample}/'

        for model in models:

            print(model)
            params = jnp.load(f'{params_in}/{dataset}/dict_param_lag_{max_lag}.pkl', allow_pickle=True)
            # if model == 'tensor':
            data = jnp.load(f'{dir_in}{model}_fig_oos_ret_rankptf_ver3/dict_{model}_oos_{start_year}.pkl', allow_pickle=True)
            # else:
            #     data = jnp.load(f'{dir_in}{model}_fig_oos_ret_rankptf_ver3/dict_{model}_oos_{start_year}.pkl', allow_pickle=True)
            if dataset == 'char_anom':
                bin_labels, _, _, max_lag, frac_longshort, all_dates, start_date_maxlag = params.values()
            else:
                bin_labels, all_dates = params['lst_char'], params['all_dates']

            # start_date_oos parameter can be changed in the _load_datasets.py file
            dates = all_dates[-len(all_dates[all_dates >= start_date_oos]) - window_size:] if oos else all_dates
            start_year = str(dates[window_size])[:4]
            start_date, end_date = str(dates[window_size])[:10], str(dates[-max_horizon - 1])[:10]
            
            # 2 plots
            if model == "tensor": 
                for method in ["TFM", "Naive"]:
                    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
                    fig.suptitle(f'{"" if method == "TFM" else method} Tensor Factor Model Multiperiod Results ({dataset.upper()})', fontsize=14)
                    sr = data[method][LAG].mean(axis=0) / data[method][LAG].std(axis=0)
                    if method == "Naive":
                        method = "Precise"
                    plot(sr, lst_K, max_horizon, method, window_size, LAG, fig, axes, start_date, end_date)

                    filename = f'{dataset}_{model}_{method}'
                    fig.savefig(f'{dir_out}{filename}', bbox_inches='tight')

            # 1 plot
            # elif model == "pooled_pca":
            #     fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            #     fig.suptitle(f'Pooled PCA Multiperiod Results ({dataset.upper()})', fontsize=14)
            #     sr = data[LAG].mean(axis=0) / data[LAG].std(axis=0)
            #     plot(sr, lst_K, max_horizon, "Pooled PCA", window_size, LAG, fig, axes, start_date, end_date)

            #     filename = f'{dataset}_{model}'
            #     fig.savefig(f'{dir_out}{filename}', bbox_inches='tight')

            # 1 plot 
            elif model == "pca":
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
                fig.suptitle(f'PCA Multiperiod Results ({dataset.upper()})', fontsize=14)
                sr_lst = []
                model_free = []
                for horizon in range(1, max_horizon + 1):
                    sr = data[gamma][horizon].mean(axis=1) / data[gamma][horizon].std(axis=1)
                    sr_lst.append(jnp.expand_dims(sr, axis=0))
                    model_free_sr = data['model_free'][horizon].mean() / data['model_free'][horizon].std() # dim: scalar
                    model_free.append(jnp.expand_dims(model_free_sr, axis=0))

                sr = jnp.concatenate(sr_lst, axis=0) # dim: (max_horizon, len(lst_K))
                temp = jnp.concatenate(model_free, axis=0) # dim: (max_horizon)
                plot_sr = jnp.concatenate([sr, temp[..., None]], axis=1)
                if dataset == 'ff':
                    lst_K = [1, 2, 3, 4]
                plot(plot_sr, lst_K + ['Model Free'], max_horizon, "PCA", window_size, 0, fig, axes, start_date, end_date)


                filename = f'{dataset}_{model}'
                fig.savefig(f'{dir_out}{filename}', bbox_inches='tight')