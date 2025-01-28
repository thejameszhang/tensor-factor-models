# Derivations 

Using the tensor model, let's derive the formulas for the means and covariances of $K$ basis assets denoted by $\mu_k^{F, S}$ and $\text{Var}_k^{F, S}$ in the paper. We will show both moments at first for any arbitrary basis asset and then expand to compute the vector of means and the covariance matrix.

## Expected Value

From the tensor model, we have that 
$$R_{t, i, l} = \sum_{k=1}^K \lambda_k F_{t, k} B_{i, k} W_{l, k}$$

Note that here $F_{t, k}$ is a scalar, but in the rest of the document, it is a random variable denoting the basis asset $k$ return at time $t$.
We include the normalizing scalar $\lambda_k$ in the loadings $B_k$ and thus drop it from the notations. Furthermore, note that all returns are log returns to simplify the aggregations over horizons. 

$$\mu_{k}^{F, S} = \mathbb{E}_t \left[\sum_{l=1}^S \left( \sum_{k=1}^K F_{t+S-l, k} B_{i, k} W_{l, k}\right) \right]$$

The basis assets are orthogonal by construction [they are actually not orthogonal in tensor decompositions unlike as in PCA!] and any asset has a beta of $1$ to itself (ie. the matrix $B$ would end up being $I_{k \times k}$) and so the above simplifies to 

$$\mu_{k}^{F, S} = \mathbb{E}_t \left[\sum_{l=1}^S F_{t+S-l, k} W_{l, k}\right]$$ 

note - I think this indexing is right? This should properly align the "future" returns of basis asset $k$ with the lag (from the perspective of time $t+S$). This doesn't change the formula but is important to clarify before implementing "the naive way."

By linearity of expectation, 
$$\mu_k^{F, S} = \sum_{l=1}^S W_{l, k} \mathbb{E}_t[F_{t + S - l, k}] = \left( \sum_{l=1}^S W_{l, k}\right)\mathbb{E}_t[F_{t, k}]$$

as desired (returns usually stationary time series). This easily translates to a vector of means
$$\mu^{F, S} = \left( \sum_{l=1}^S W_k \right) \odot \mathbb{E}_t [F_t]$$
where $\odot$ is element-wise multiplication and $\mu^{F, S} \in \mathbb{R}^{K}$. 

## Variance

(*) If returns for different months are uncorrelated,

$$\text{Var}_k^{F, S} = \text{Var}_t \left[ \sum_{l=1}^S F_{t + S- l, k} W_{l, k}\right]$$
$$\text{Var}_k^{F, S} = \left(\sum_{l=1}^S W_{l, k}^2 \text{Var}[F_{t + S - l, k}]\right) + 2\sum_{i \neq j}W_{i, k}W_{j, k}\text{Cov}(F_{t + i}, F_{t+j})$$
Since returns for different months are uncorrelated, the covariance terms cancel, and this leaves us with 
$$\text{Var}_k^{F, S} = \left(\sum_{l=1}^S W_{l, k}^2\right)\text{Var}_t[F_{t, k}]$$

[You should also show this for covariances of factors (obvious), since factors are not orthogonal]

For covariances, 
$$\text{Cov}_{i, j}^S\left( \sum_{l=1}^S F_{t + S - l, i} W_{l, i}, \sum_{l=1}^S F_{t+S-l, j}W_{l, j}\right) = $$

It can be easily seen that this translates to 
$$\text{Var}^{F, S} = \left( \sum_{l=1}^S W_l W_l^\top\right)\odot \text{Var}_t(F_t) \in \mathbb{R}^{K \times K}$$

# Multiperiod PCA Benchmark Model

Observe the `PCA_Multiperiod_Unmapped` function. Let $t$ = `idx_window` which represents the earliest date in the rolling window. Furthermore, let $W$ = `window_size`, $h$ = `horizon`. That is, $[t, t + W - 1]$ is the date range in the window, and $[t+W, t+W+h - 1]$ is the OOS period in our rolling window approach. To provide some intution, we are an investor at month $t + W - 1$ and want to optimize our portfolio over an $h$-month horizon. 

$X_{IS} = X[t: t + W, :h, :] \in \mathbb{R}^{W \times h \times N}$ after slicing. We then wish to calculate $h$-month returns returns, which I will denote $X_{IS}^h \in \mathbb{R}^{W - h + 1, N}$. That is, for all $t \in [0, W - h]$ and $i \in 1 \ldots N$, 

$$\mu_i^h = \sum_{l=0}^{h-1} X[t+l, l, i] \Longleftrightarrow \sum_{l=1}^h R_{t+l, i, l} = \sum_{l=1}^h R_{t+l, i} | C_{t, i}$$

where the $\Longleftrightarrow$ represents a transition from $0$-indexing to $1$-indexing. Thus, the entry $(t, i) \in X_h$ represents the $h$-month buy-and-hold return on characteristic-based factor $i$ conditioned on information at month $t$. An example - suppose $t = 5, h = 2$. Then, in code, we would be computing $X[5, 0, :] + X[6, 1, :]$ which after converting from the $0$-indexing to the $1$-indexing used in our mathematical notation, would be equivalent to $R_6 | C_5 + R_7 | C_5$. This matrix $X_h$ is computed in the function `get_multiperiod_returns`.

From there, use $X_h$ as input to RP-PCA to obtain basis assets $\hat{F} \in \mathbb{R}^{(W - h + 1) \times K}$ and loadings $\hat{\Lambda} \in \mathbb{R}^{N \times K}$. Importantly, the basis assets are $h$-month returns. Now, we can easly compute basis asset means $\mu \in \mathbb{R}^K$, basis asset covariance matrix $\Sigma \in \mathbb{R}^{K \times K}$ using a Newey-West Covariance estimator due to the fact that $X_h$ is a matrix of rolling returns, and SDF weights.

To test our estimators, we take $X_{OOS} = X[t + W: t + W + h, :h, :] \in \mathbb{R}^{h \times h \times N}$. We use the same `get_multiperiod_returns` function to compute the matrix $X^h_{OOS} \in \mathbb{R}^N$, which represents the realized $h$-month horizon return of the $N$ characteristic-based factors. The OOS basis asset values are recovered from a regression of returns on the estimated loadings

$$\hat{F}_{OOS} = X_{OOS}^h \hat{\Lambda} (\hat{\Lambda}^T \hat{\Lambda})^{-1} \in \mathbb{R}^K$$

and combined with our SDF weights, $\langle \hat{F}_{OOS}, \Sigma^{-1} \mu \rangle$ yields our returns.

# Pooled PCA Benchmark Model




