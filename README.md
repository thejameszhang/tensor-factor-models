# Term Structures of Firm Characteristics and Multi-Horizon Investments

See the slides or preliminary paper for a brief introduction to the motivation and technical specifications of our model. 

## Datasets

1. Characteristic Anomalies Dataset (factors are long-short decile portfolios)
2. Fama French 5 + Momentum Dataset
3. Shrinking the Cross-Section Dataset
4. Toy Dataset (only value portfolios)
5. WRDS Dataset (100+ factors)


## File Structure Breakdown

```
├── data
│   └── characteristics
│       └── 01DEC20
│           ├── raw
│           ├── std
│           └── stdrank
├── interpretability
│   └── figures
├── notebooks
├── notes
│   ├── figures
│   │   ├── full
│   │   └── post-2005
│   └── helpers
├── organized_data
│   └── organized_data
│       ├── char_anom
│       ├── ff
│       ├── scs
│       ├── toy
│       └── wrds
├── orig
│   └── __pycache__
├── results_oos
│   ├── multiperiod
│   │   ├── char_anom
│   │   │   ├── fig_onefit_oos_ret_rankptf_ver1
│   │   │   ├── fig_oos_ret_rankptf_ver2
│   │   │   ├── pca_fig_oos_ret_rankptf_ver3
│   │   │   ├── tensor_fig_oos_ret_rankptf_ver3
│   │   │   └── tensor_fig_oos_ret_rankptf_ver4
│   │   ├── ff
│   │   │   ├── pca_fig_oos_ret_rankptf_ver3
│   │   │   ├── tensor_fig_oos_ret_rankptf_ver3
│   │   │   └── tensor_fig_oos_ret_rankptf_ver5
│   │   ├── scs
│   │   │   ├── fig_onefit_oos_ret_rankptf_ver1
│   │   │   ├── fig_onefit_oos_ret_rankptf_ver2
│   │   │   ├── fig_oos_ret_rankptf_ver2
│   │   │   ├── lasso
│   │   │   │   ├── betas
│   │   │   │   └── lags
│   │   │   ├── pca_fig_oos_ret_rankptf_ver2
│   │   │   ├── pca_fig_oos_ret_rankptf_ver3
│   │   │   ├── tensor_fig_oos_ret_rankptf_ver3
│   │   │   ├── tensor_fig_oos_ret_rankptf_ver4
│   │   │   └── tensor_fig_oos_ret_rankptf_ver5
│   │   ├── toy
│   │   │   ├── tensor_fig_oos_ret_rankptf_ver3
│   │   │   └── tensor_fig_oos_ret_rankptf_ver4
│   │   └── wrds
│   │       ├── fig_onefit_oos_ret_rankptf_ver1
│   │       ├── fig_oos_ret_rankptf_ver2
│   │       ├── pca_fig_oos_ret_rankptf_ver3
│   │       ├── tensor_fig_oos_ret_rankptf_ver3
│   │       └── tensor_fig_oos_ret_rankptf_ver4
│   ├── one_fit
│   │   ├── char_anom
│   │   │   ├── fig_onefit_oos_ret_rankptf_ver1
│   │   │   │   ├── 120
│   │   │   │   └── 60
│   │   │   ├── fig_onefit_oos_ret_rankptf_ver2
│   │   │   │   └── 60
│   │   │   ├── tbl_onefit_oos_ret_rankptf_ver1
│   │   │   │   └── 60
│   │   │   └── tbl_onefit_oos_ret_rankptf_ver2
│   │   │       └── 60
│   │   ├── scs
│   │   │   ├── fig_onefit_oos_ret_rankptf_ver1
│   │   │   │   ├── 120
│   │   │   │   └── 60
│   │   │   ├── fig_onefit_oos_ret_rankptf_ver2
│   │   │   │   └── 60
│   │   │   ├── fig_onefit_oos_ret_rankptf_ver4
│   │   │   │   └── 120
│   │   │   ├── tbl_onefit_oos_ret_rankptf_ver1
│   │   │   │   ├── 120
│   │   │   │   └── 60
│   │   │   ├── tbl_onefit_oos_ret_rankptf_ver2
│   │   │   │   └── 60
│   │   │   └── tbl_onefit_oos_ret_rankptf_ver4
│   │   │       └── 120
│   │   └── wrds
│   │       ├── fig_onefit_oos_ret_rankptf_ver1
│   │       │   ├── 120
│   │       │   └── 60
│   │       └── tbl_onefit_oos_ret_rankptf_ver1
│   │           ├── 120
│   │           └── 60
│   └── pca_loadings
│       ├── char_anom
│       ├── scs
│       └── wrds
├── scripts
├── tests
└── tfm
    ├── config
    ├── models
    └── utils
```

Version 1 is the original versions. Version 2 is based on Jax code that I wrote. Version 3 is optimizing it using faster Jax functions. Version 4 correctly implements fixed intercept modes and does not impose orthogonality in any modes. Version 5 is the orthogonal model. 

## LaTeX Files
interpretability/ - focused on economic content and interpreting the lag loadings dimension

notes/ - focused on the results of the Tensor Factor Model as a pricing model ie. unexplained variation, CAPM alphas, etc.

orthogonality/ - focused on comparing the orthogonal and non-orthogonal models