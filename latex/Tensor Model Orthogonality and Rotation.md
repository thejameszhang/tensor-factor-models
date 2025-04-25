Okay, here is the updated Markdown report incorporating the inline citations and the Works Cited list from the Google Doc version.

***

# Analysis of Orthogonality and Rotational Freedom in the PARAFAC/CP Tensor Factor Model for Asset Returns

## 1. Introduction

### 1.1. Context: Factor Models and Term Structure in Asset Pricing

Factor models are foundational tools in modern asset pricing, aiming to explain the complex dynamics of asset returns through exposure to a smaller set of underlying, often unobserved, risk factors. Traditional approaches, largely based on matrix factorizations like Principal Component Analysis (PCA) or statistical factor analysis, typically model the covariance structure of returns across assets at a single point in time or the time-series evolution of returns for individual assets. However, financial data often possesses a richer, multi-modal structure. Your research addresses this by studying the term structure of characteristic-sorted portfolios, specifically modeling monthly expected excess returns conditional on characteristics observed at various lags ($l=1..L$). This involves analyzing a three-dimensional data array (tensor) of returns. Tensor factor models offer a powerful framework to decompose these multi-way datasets, potentially uncovering latent factors that drive returns across multiple dimensions simultaneously, thereby offering deeper insights into the underlying economic mechanisms.

### 1.2. Model Specification: The PARAFAC/CP Decomposition

You propose a novel "tensor factor model" to provide a parsimonious framework for this term structure. The core of your model is the decomposition of the return tensor $\mathcal{R} \in \mathbb{R}^{T \times N \times L}$ (Time $\times$ Asset $\times$ Lag) using the following structure:
$$r_{t,i,l} = \sum_{k=1}^K f_{t,k} \cdot b_{i,k} \cdot g_{l,k}$$
Here, $f_{t,k}$ represents the time-series of factor $k$, $b_{i,k}$ is the loading of asset $i$ on factor $k$, and $g_{l,k}$ captures the loading profile across the lag dimension $l$ for factor $k$. (Note: Your notation uses $F, B, W$ for the matrices and $f, b, g$ or $w$ for the vectors/elements, which I will adopt. The scalar $\lambda_k$ is noted as being subsumed into the loadings).

This mathematical structure is the standard **Canonical Polyadic Decomposition (CPD)**.[1] It is widely known as **PARAFAC** (Parallel Factor Analysis) or **CANDECOMP** (Canonical Decomposition).[1] The model was independently proposed by Harshman (1970) under the name PARAFAC and by Carroll and Chang (1970) as CANDECOMP.[1] The model decomposes the tensor $\mathcal{R}$ into a sum of $K$ rank-one tensors [8]:
$$\mathcal{R} \approx \sum_{k=1}^K f_k \circ b_k \circ w_k$$
where $f_k \in \mathbb{R}^T$, $b_k \in \mathbb{R}^N$, $w_k \in \mathbb{R}^L$ are the columns of the factor matrices $F \in \mathbb{R}^{T \times K}$, $B \in \mathbb{R}^{N \times K}$, and $W \in \mathbb{R}^{L \times K}$, and '$\circ$' denotes the vector outer product. The minimal number $K$ for an exact decomposition is the tensor rank.[1] Your work uses the PARAFAC algorithm, typically based on Alternating Least Squares (ALS), to estimate this model.[2, 9, 10]

### 1.3. Report Objectives and Scope

This report analyzes two critical mathematical properties of this specific PARAFAC/CP model:
1.  Can orthogonality be imposed on the factor matrices ($F, B, W$) without loss of generality?
2.  What are the consequences of imposing orthogonality?
3.  Does the model possess rotational freedom, and is factor rotation a valid operation?

The analysis focuses on the inherent mathematical properties of the standard PARAFAC/CP structure defined above. This distinguishes it from related but distinct models such as PARAFAC2, which allows one mode's factor matrix to vary across slices under specific constraints [15], or other constrained tensor models that incorporate specific structures like block-diagonal core tensors or orthogonality in specific modes by design.[4] The properties discussed herein pertain specifically to the model $r_{t,i,l} = \sum_{k=1}^K f_{t,k} \cdot b_{i,k} \cdot w_{l,k}$.

## 2. The PARAFAC/CP Decomposition: Structure and Relation to Other Models

### 2.1. Formal Definition and Notation

The PARAFAC/CP model approximates the return tensor $\mathcal{R} \in \mathbb{R}^{T \times N \times L}$ as:
$$r_{t,i,l} = \sum_{k=1}^K f_{t,k} \cdot b_{i,k} \cdot w_{l,k} + e_{t,i,l}$$
where $e_{t,i,l}$ is the residual error. The goal is to find factor matrices $F = [f_1, \dots, f_K] \in \mathbb{R}^{T \times K}$, $B = [b_1, \dots, b_K] \in \mathbb{R}^{N \times K}$, and $W = [w_1, \dots, w_K] \in \mathbb{R}^{L \times K}$ that minimize the sum of squared residuals, $\sum_{t,i,l} e_{t,i,l}^2$, typically via Alternating Least Squares (ALS).[2, 5, 10, 11] The columns of these matrices are the factor vectors.[11]

Using slice notation, the $l$-th slice (holding lag constant) $\mathcal{R}_{::l} \in \mathbb{R}^{T \times N}$ is approximated by [3, 5, 11]:
$$\mathcal{R}_{::l} \approx F \text{diag}(w_{l,:}) B^T$$
where $w_{l,:} = [w_{l,1}, \dots, w_{l,K}]$ is the $l$-th row of $W$. Your alternative representation $\overrightarrow{R} = F \Lambda^{\top}$ where $\Lambda = W \odot B$ (Khatri-Rao product) highlights the connection to PCA on the unfolded tensor but emphasizes the restricted structure of $\Lambda$ imposed by the PARAFAC/CP model. The smallest integer K for which the decomposition holds exactly (i.e., E=0) is the rank of the tensor $\mathcal{R}$.[1]

### 2.2. Contrast with Tucker Decomposition

The Tucker decomposition is a more general tensor model [3, 6, 10, 12, 13, 14]:
$$\mathcal{R} \approx \mathcal{G} \times_1 F \times_2 B \times_3 W$$
It involves factor matrices $F, B, W$ (potentially with different numbers of columns $P, Q, R$) and a core tensor $\mathcal{G} \in \mathbb{R}^{P \times Q \times R}$ that governs the interactions between components.[2, 3, 12, 13, 15, 16, 26] PARAFAC/CP is a special case where $P=Q=R=K$ and the core tensor $\mathcal{G}$ is superdiagonal.[2, 3, 8, 13] The absence of a general core tensor in PARAFAC/CP restricts interactions – factor $f_k$ only interacts with $b_k$ and $w_k$ – which is fundamental to its uniqueness properties.[2, 16, 17, 18] The core tensor in the Tucker model allows for interactions between any component p from mode 1, component q from mode 2, and component r from mode 3.[26]

### 2.3. Contrast with Principal Component Analysis (PCA)

As you note, standard PCA can be applied to the unfolded tensor $\overrightarrow{R} \in \mathbb{R}^{T \times NL}$. PCA finds orthogonal factors $F^{\text{PCA}}$ and loadings $\Lambda^{\text{PCA}}$.[5, 18, 19, 20] Your tensor model $r_{t,i,l} = \sum f b w$ imposes a specific structure on the loadings matrix $\Lambda = (W \odot B)^T$, making it a restricted version of PCA. Key differences include:
* PCA operates on the unfolded matrix, potentially obscuring the three-way interactions PARAFAC/CP explicitly models.[9, 18, 21]
* PCA factors/components are orthogonal by definition [5]; PARAFAC/CP factors are generally not.[7, 9, 22, 23, 24]
* PCA solutions have rotational freedom; PARAFAC/CP solutions are typically unique (discussed below).[5, 7, 9, 18, 20, 21, 25-33]

## 3. Orthogonality Constraints in PARAFAC/CP

### 3.1. The Question of Orthogonality

Can orthogonality constraints (e.g., $F^T F = I_K$, $B^T B = I_K$, or $W^T W = I_K$) be imposed on the PARAFAC/CP factor matrices without loss of generality?

### 3.2. Lack of Generality in Standard PARAFAC/CP

For the standard PARAFAC/CP model $r_{t,i,l} = \sum f b w$, the answer is generally **no**.[7] Imposing orthogonality on any factor matrix restricts the model's ability to represent arbitrary tensors and typically leads to a loss of fit compared to the unconstrained model.[7, 10, 16, 17, 20, 22, 23, 34] The model structure does not inherently require or guarantee orthogonality.[7, 9, 22, 23] This is because PARAFAC/CP lacks the rotational freedom (unlike Tucker/HOSVD [26]) needed to transform factors to an orthogonal basis while preserving the fit.[2, 13, 16, 24] The rotational freedom inherent in the Tucker model allows orthogonality to be achieved, with adjustments absorbed by the core tensor $\mathcal{G}$.[24] PARAFAC/CP lacks this core tensor and associated rotational freedom.[24] Forcing orthogonality requires the tensor itself to have a special, rarely encountered structure (akin to diagonalizability).[16, 26]

### 3.3. Consequences of Forcing Orthogonality

If orthogonality is forced (e.g., algorithmically), especially when the true underlying factors (e.g., economic drivers) are correlated:

1.  **Reduced Goodness-of-Fit:** The constrained model will likely yield a higher residual error (poorer fit) because it cannot capture the optimal non-orthogonal structure.[10, 16, 20] The model's ability to capture the data's structure is compromised.
2.  **Misrepresentation of Factors:** The estimated factors become mathematical artifacts forced to be orthogonal, rather than representing the true, potentially correlated, underlying sources.[10, 20] This severely hinders meaningful interpretation.
3.  **Interpretability Issues:** While orthogonal factors can seem simpler, forcing them can obscure true correlations and lead to misleading interpretations of the underlying system.[10, 20, 26] The "simplicity" may not reflect reality.
4.  **Degeneracy:** Orthogonality constraints are sometimes used algorithmically to combat *degeneracy* (highly correlated factors with diverging magnitudes).[10, 15, 20, 22, 23, 25, 26, 35, 36] However, this is a numerical fix that imposes potentially incorrect structure and should not be confused with the model's inherent properties.[10, 20, 22, 23, 25] It forces a potentially incorrect structure onto the data simply to avoid an algorithmic pathology.[20]

### 3.4. Orthogonalization in Algorithms (e.g., Orth-ALS)

Algorithms like Orthogonalized Alternating Least Squares (Orth-ALS) use orthogonalization steps *during* iteration to improve convergence and stability, especially when factors are correlated or weights are non-uniform.[19, 36, 37, 38] The motivation is primarily algorithmic: to prevent multiple estimated factor vectors from converging to the same underlying true factor and avoid poor local optima.[19] This is an *algorithmic* tool to help find the standard (potentially non-orthogonal) PARAFAC/CP solution more reliably; it does not change the final model structure or guarantee orthogonal output factors.[19, 36, 38] The final factor matrices obtained from applying Orth-ALS to a general tensor are not guaranteed to be orthogonal. Attempting to guarantee recovery of non-orthogonal tensors using algorithms designed for orthogonal tensors typically requires a "whitening" preprocessing step, which can be complex.[37]

### 3.5. Specific Constrained Models

Variants like PARAFAC with orthogonality in one mode [2, 38, 39, 40] or PARAFAC2 [2, 11, 15, 38, 41] exist but are distinct from the standard model. They incorporate orthogonality by design for specific applications or data structures, sometimes leading to milder conditions for uniqueness.[38] PARAFAC2 involves constraints related to the constancy of the cross-product matrix, implicitly involving orthogonality considerations.[15] These do not alter the property that for standard PARAFAC/CP, forcing orthogonality is generally a restrictive constraint.

## 4. Essential Uniqueness of the PARAFAC/CP Model

A key strength of PARAFAC/CP is its *essential uniqueness* under relatively mild conditions.[2-5, 7, 8, 26, 32, 36, 40, 42-49] This property fundamentally distinguishes it from matrix factorizations like PCA and SVD, and from the more general Tucker tensor decomposition.[4]

### 4.1. Definition of Essential Uniqueness

Essential uniqueness means a PARAFAC/CP decomposition is unique up to:
1.  **Permutation:** The order of the $K$ components $(f_k, b_k, w_k)$ can be changed.[3, 4, 5, 7, 8]
2.  **Scaling:** Factor vectors within a component $k$ can be scaled ($f_k \rightarrow \alpha f_k$, $b_k \rightarrow \beta b_k$, $w_k \rightarrow \gamma w_k$) provided $\alpha \beta \gamma = 1$.[3, 4, 5, 7, 8]

Formally, if $\mathcal{R} = \sum_{k=1}^K f_k \circ b_k \circ w_k = \sum_{k=1}^K \tilde{f}_k \circ \tilde{b}_k \circ \tilde{w}_k$ are two rank-$K$ PARAFAC/CP decompositions of $\mathcal{R}$ satisfying uniqueness conditions, then there exists a unique permutation matrix $\Pi$ and unique diagonal scaling matrices $\Lambda_F, \Lambda_B, \Lambda_W$ such that the factor matrices are related by [3-5, 8, 11]:
$\tilde{F} = F \Pi \Lambda_F$, $\tilde{B} = B \Pi \Lambda_B$, $\tilde{W} = W \Pi \Lambda_W$ 
with the constraint that $\Lambda_F \Lambda_B \Lambda_W = I_K$. These are considered "trivial" indeterminacies.

### 4.2. Kruskal's Condition for Uniqueness

Kruskal's theorem (1977) provides a widely cited *sufficient* condition for essential uniqueness.[2-5, 8, 40, 42, 44-46, 48] It uses the *k-rank* of a matrix (denoted $k_A$), which is the maximum integer $k$ such that *every* set of $k$ columns is linearly independent.[3-5, 8, 10] ($k_A \le \text{rank}(A)$).[10]

Kruskal's condition states that uniqueness is guaranteed if:
$$k_F + k_B + k_W \ge 2K + 2$$
where $K$ is the number of components.[2-5, 8] This requires sufficient linear independence among the columns of the factor matrices relative to the number of components. Verifying the k-rank can be computationally demanding.[8]

### 4.3. Sufficiency vs. Necessity

Kruskal's condition is sufficient, but **not always necessary** for uniqueness when $K \ge 4$.[8, 40-46, 48] Uniqueness might hold even if the condition is violated.[8] For $K=2$ and $K=3$, the condition is both necessary and sufficient.[8, 40-42, 44-46, 48] A conjecture linking k-rank to rank for necessity when $K>3$ was refuted.[41] Relying solely on Kruskal's condition might lead to incorrectly concluding non-uniqueness for $K \ge 4$.

### 4.4. Weaker Conditions

Other sufficient conditions exist, often involving rank properties of Khatri-Rao products (e.g., $F \odot B$) [3, 4, 5, 40, 42] or requiring at least one factor matrix to have full column rank.[3, 4, 5, 21, 40] Research also explores partial uniqueness [39] and uniqueness of constrained variants.[38]

## 5. Implications for Factor Rotation

### 5.1. Absence of Rotational Freedom

Essential uniqueness directly implies a **lack of rotational freedom** in PARAFAC/CP.[5, 7, 9-10, 18-21, 24-33, 35, 44, 45] Unlike PCA or Tucker, where transformations (rotations) can be applied without changing the model fit [5, 9, 18, 20, 21, 25-33], any transformation of PARAFAC/CP factors beyond permutation and scaling will alter the decomposition and worsen the fit.[7, 9, 10, 18, 35, 44] The model's rigid structure, lacking an intermediary like the Tucker core tensor to absorb rotations, prevents such transformations.[2, 19, 24] In PCA, $X \approx AB^T = (AQ)(BQ)^T$ for orthogonal $Q$.[30] In Tucker, $\mathcal{R} \approx \mathcal{G} \times_1 F \times_2 B \times_3 W = (\mathcal{G} \times_1 Q^T) \times_1 (FQ) \times_2 B \times_3 W$.[5]

### 5.2. Invalidity of Post-Hoc Factor Rotation

Applying rotation techniques (Varimax, Quartimax, Promax, etc.) to PARAFAC/CP factors ($F, B, W$) is **mathematically invalid and conceptually inappropriate**.[5, 7, 9-10, 18, 21, 26, 27, 31, 35, 44, 45] These methods are designed for models *with* rotational ambiguity to achieve simple structure.[5] Since PARAFAC/CP aims for a unique solution, rotation is unnecessary and detrimental to the model's fit and structural integrity. Applying rotation forces factors into a configuration violating the PARAFAC/CP structure and no longer providing the best least-squares fit. PARAFAC was developed partly to *avoid* the rotation problem of two-way factor analysis.[5, 25, 26, 31, 44]

### 5.3. Uniqueness as an Advantage

The lack of rotational freedom is a major advantage.[7, 9-10, 15, 18, 21, 26-28, 31-33, 35, 42-45] If the model is appropriate and unique (checked via fit, residuals, stability, core consistency [7]), the estimated factors potentially represent meaningful, intrinsically defined latent structures, allowing for direct interpretation without subjective rotation choices.[7, 9-10, 15, 18, 21, 26-28, 32, 33, 35, 44, 45] This potential for direct identification of "explanatory" factors [45] makes PARAFAC/CP appealing in scientific domains.[7]

## 6. Interpretation and Contextual Relevance for Asset Pricing

### 6.1. Direct Factor Interpretation

If the PARAFAC/CP model fits your return tensor well and uniqueness holds, the factors $f_k, b_k, w_k$ can potentially be interpreted directly.[5, 7, 9, 10, 18, 35, 44, 45] After resolving scaling ambiguity via normalization (e.g., unit norm columns for two modes, or setting $w_{1,k}=1$ as you mentioned) [7], the factors represent:
* $f_k$: Time-series dynamics of latent factor $k$.
* $b_k$: Cross-sectional asset exposures to factor $k$.
* $w_k$: Lag profile or term structure of factor $k$'s influence.

### 6.2. Identifying Latent Economic Drivers

The unique $(f_k, b_k, w_k)$ triplets may correspond directly to distinct economic mechanisms driving the term structure of returns.[5, 15, 26, 45] This contrasts with PCA where economic meaning is often sought *after* rotation. PARAFAC/CP attempts to find "explanatory" factors inherent in the data's multi-way structure.[45]

### 6.3. Handling Correlated Factors

PARAFAC/CP does not require factors to be orthogonal, allowing it to model potentially correlated economic drivers, which is often more realistic than PCA's enforced orthogonality.[10, 20, 42] It can recover correlated factors provided their patterns of influence across modes are distinct enough for uniqueness.

### 6.4. Caveats: Model Appropriateness and Degeneracy

* **Model Appropriateness:** Interpretation hinges on the PARAFAC/CP model (sum of trilinear components) being a good representation of the data's structure.[7] Validation (residuals, fit, stability [7], core consistency [7, 9]) is crucial.[7, 9, 15, 25, 26, 45] Diagnostics like CORCONDIA help assess the structure and chosen K.[7]
* **Degeneracy:** ALS estimation can sometimes yield degenerate solutions (highly correlated, diverging factors).[10, 15, 20, 22, 23, 25, 26, 35, 36] This often indicates model misspecification (e.g., $K$ too large) or poor data fit.[20] Careful handling (model selection, regularization) is needed rather than simply forcing orthogonality.[10, 20, 22, 23, 25]

## 7. Synthesis and Conclusion

### 7.1. Summary of Findings

For the PARAFAC/CP model $r_{t,i,l} = \sum_{k=1}^K f_{t,k} \cdot b_{i,k} \cdot w_{l,k}$:
1.  **Orthogonality:** Cannot generally be imposed on factor matrices ($F, B, W$) without loss of generality or fit.[7, 10, 16, 17, 20, 22, 23, 34] The model accommodates non-orthogonal factors.
2.  **Uniqueness:** The decomposition is essentially unique (up to permutation and scaling) under conditions like Kruskal's ($k_F + k_B + k_W \ge 2K + 2$).[2-5, 8, 40, 42, 44-46, 48]
3.  **Rotation:** Due to uniqueness, the model lacks rotational freedom. Applying factor rotation techniques is invalid and inappropriate.[5, 7, 9-10, 18, 21, 26, 27, 31, 35, 44, 45]

### 7.2. Implications for Your Research

* Your tensor model (PARAFAC/CP) provides a framework for potentially identifying **unique latent factors** driving the term structure of asset returns across time, assets, and lags.
* If the model fits well and uniqueness holds, you can aim for **direct interpretation** of the estimated factors $f_k, b_k, w_k$ (after normalization) without needing rotation.
* The model's ability to handle **correlated factors** is suitable for economic data.
* You should **avoid imposing orthogonality** unless using a specific constrained variant, and **do not apply factor rotation** to the results.
* Focus on **model validation** (fit, residuals, stability, core consistency) and checking for **degeneracy** to ensure reliable and interpretable results.

In conclusion, the PARAFAC/CP model, characterized by its essential uniqueness and consequent lack of rotational freedom, offers a distinct and potentially advantageous approach for analyzing multi-way asset return data compared to traditional matrix methods or more flexible tensor models like Tucker. Its strength lies in the potential to directly identify unique, interpretable latent structures, provided the model assumptions align with the data characteristics and careful validation is performed.

***

## Works cited

1.  Tensor rank decomposition - Wikipedia, accessed April 17, 2025, [https://en.wikipedia.org/wiki/Tensor_rank_decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition)
2.  CANDECOMP/PARAFAC Decomposition of High-order Tensors Through Tensor Reshaping - arXiv, accessed April 17, 2025, [https://arxiv.org/pdf/1211.3796](https://arxiv.org/pdf/1211.3796)
3.  Bayesian Conditional Tensor Factorizations for High-Dimensional Classification - PMC, accessed April 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6980791/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6980791/)
4.  Overview of Constrained PARAFAC Models - arXiv, accessed April 17, 2025, [http://arxiv.org/pdf/1405.7442](http://arxiv.org/pdf/1405.7442)
5.  Three-Way Component Analysis Using the R Package ThreeWay, accessed April 17, 2025, [https://www.jstatsoft.org/article/view/v057i07/738](https://www.jstatsoft.org/article/view/v057i07/738)
6.  Scalable tensor factorizations for incomplete data - Sandia National Laboratories, accessed April 17, 2025, [https://www.sandia.gov/app/uploads/sites/143/2021/10/daniel-dunlavy-2011-AcDuKoMo11.pdf](https://www.sandia.gov/app/uploads/sites/143/2021/10/daniel-dunlavy-2011-AcDuKoMo11.pdf)
7.  2. Basic PARAFAC modeling - Chemometrics Research, accessed April 17, 2025, [https://ucphchemometrics.com/2-basic-parafac-modeling/](https://ucphchemometrics.com/2-basic-parafac-modeling/)
8.  (PDF) On Uniqueness in Candecomp/Parafac - ResearchGate, accessed April 17, 2025, [https://www.researchgate.net/publication/225107126_On_uniqueness_in_CandecompParafac](https://www.researchgate.net/publication/225107126_On_uniqueness_in_CandecompParafac)
9.  Speeding up PARAFAC - DiVA portal, accessed April 17, 2025, [http://www.diva-portal.org/smash/get/diva2:1216742/FULLTEXT01.pdf](http://www.diva-portal.org/smash/get/diva2:1216742/FULLTEXT01.pdf)
10. www.kolda.net, accessed April 17, 2025, [https://www.kolda.net/publication/TensorReview.pdf](https://www.kolda.net/publication/TensorReview.pdf)
11. On Kruskal's uniqueness condition for the Candecomp/Parafac decomposition - Department of Electrical and Computer Engineering, accessed April 17, 2025, [http://www.ece.umn.edu/~nikos/SteSidLAAreprint.pdf](http://www.ece.umn.edu/~nikos/SteSidLAAreprint.pdf)
12. Tensor Decompositions and Applications - ResearchGate, accessed April 17, 2025, [https://www.researchgate.net/publication/220116494_Tensor_Decompositions_and_Applications](https://www.researchgate.net/publication/220116494_Tensor_Decompositions_and_Applications)
13. Tensor Decompositions and Applications | SIAM Review, accessed April 17, 2025, [https://epubs.siam.org/doi/10.1137/07070111X](https://epubs.siam.org/doi/10.1137/07070111X)
14. A Brief Introduction to Low-Rank Tensor Decompositions - Joseph Nakao, accessed April 17, 2025, [https://jhknakao.github.io/files/tensornotes/lowranklecture_60min.pdf](https://jhknakao.github.io/files/tensornotes/lowranklecture_60min.pdf)
15. Probabilistic PARAFAC2 - MDPI, accessed April 17, 2025, [https://www.mdpi.com/1099-4300/26/8/697](https://www.mdpi.com/1099-4300/26/8/697)
16. Probabilistic PARAFAC2 - Mikkel N. Schmidt, accessed April 17, 2025, [http://www.mikkelschmidt.dk/papers/jorgensen2024entropy.pdf](http://www.mikkelschmidt.dk/papers/jorgensen2024entropy.pdf)
17. APTION: Constrainted Coupled CP And PARAFAC2 Tensor Decomposition - Computer Science and Engineering, accessed April 17, 2025, [https://www.cs.ucr.edu/~epapalex/papers/2020_ASONAM_CAPTION.pdf](https://www.cs.ucr.edu/~epapalex/papers/2020_ASONAM_CAPTION.pdf)
18. The constrained Block-PARAFAC decomposition, accessed April 17, 2025, [http://www.ece.umn.edu/~nikos/TRICAP2006main/TRICAP2006_Almeida.pdf](http://www.ece.umn.edu/~nikos/TRICAP2006main/TRICAP2006_Almeida.pdf)
19. Orthogonalized ALS: A Theoretically Principled Tensor Decomposition Algorithm for Practical Use - Proceedings of Machine Learning Research, accessed April 17, 2025, [http://proceedings.mlr.press/v70/sharan17a/sharan17a.pdf](http://proceedings.mlr.press/v70/sharan17a/sharan17a.pdf)
20. www.dss.uniroma1.it, accessed April 17, 2025, [https://www.dss.uniroma1.it/en/system/files/pubblicazioni/17-rt-giordani-14-2011.pdf](https://www.dss.uniroma1.it/en/system/files/pubblicazioni/17-rt-giordani-14-2011.pdf)
21. Kruskal's uniqueness condition for Candecomp/Parafac, accessed April 17, 2025, [https://www.ece.umn.edu/~nikos/TRICAP2006main/Stegeman-tricapkruskaltalk.pdf](https://www.ece.umn.edu/~nikos/TRICAP2006main/Stegeman-tricapkruskaltalk.pdf)
22. On uniqueness in candecomp/parafac - The Three-Mode Company, accessed April 17, 2025, [https://three-mode.leidenuniv.nl/pdf/t/tenberge2002a_pmet.pdf](https://three-mode.leidenuniv.nl/pdf/t/tenberge2002a_pmet.pdf)
23. Tensor Decompositions and Applications | SIAM Review - CS@Cornell, accessed April 17, 2025, [https://www.cs.cornell.edu/courses/cs6241/2020sp/readings/Kolda-Bader-2009-survey.pdf](https://www.cs.cornell.edu/courses/cs6241/2020sp/readings/Kolda-Bader-2009-survey.pdf)
24. Algorithms for Sparse Non-negative Tucker decompositions, accessed April 17, 2025, [https://www2.imm.dtu.dk/pubdb/edoc/imm4658.pdf](https://www2.imm.dtu.dk/pubdb/edoc/imm4658.pdf)
25. 8. TUCKER INTRODUCTION - Chemometrics Research, accessed April 17, 2025, [https://ucphchemometrics.com/8-tucker-introduction/](https://ucphchemometrics.com/8-tucker-introduction/)
26. Tucker3-ALS algorithm with orthogonality constraints on the component... - ResearchGate, accessed April 17, 2025, [https://www.researchgate.net/figure/Tucker3-ALS-algorithm-with-orthogonality-constraints-on-the-component-matrices-ALS_fig7_224317206](https://www.researchgate.net/figure/Tucker3-ALS-algorithm-with-orthogonality-constraints-on-the-component-matrices-ALS_fig7_224317206)
27. Tensor Decompositions for Large-Scale Data Mining: Methods for Uncovering Latent Patterns in Multidimensional Big Data, accessed April 17, 2025, [https://questsquare.org/index.php/JOURNALBACC/article/download/28/36/72](https://questsquare.org/index.php/JOURNALBACC/article/download/28/36/72)
28. Application of Parallel Factor Analysis (PARAFAC) to electrophysiological data - PMC, accessed April 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4311613/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4311613/)
29. www.cs.cmu.edu, accessed April 17, 2025, [http://www.cs.cmu.edu/~./pmuthuku/mlsp_page/lectures/Parafac.pdf](http://www.cs.cmu.edu/~./pmuthuku/mlsp_page/lectures/Parafac.pdf)
30. Comparing Independent Component Analysis and the Parafac model for artificial multi-subject fMRI data - Alwin Stegeman, accessed April 17, 2025, [http://www.alwinstegeman.nl/docs/Stegeman%20-%20parafac%20&%20ica.pdf](http://www.alwinstegeman.nl/docs/Stegeman%20-%20parafac%20&%20ica.pdf)
31. Introduction to Multiway Analysis - Eigenvector Research, accessed April 17, 2025, [http://www.eigenvector.com/Docs/EigenU_Europe_17/6b_Multiway_Analysis.pdf](http://www.eigenvector.com/Docs/EigenU_Europe_17/6b_Multiway_Analysis.pdf)
32. Application of Parallel Factor Analysis (PARAFAC) to the Regional Characterisation of Vineyard Blocks Using Remote Sensing Time Series - MDPI, accessed April 17, 2025, [https://www.mdpi.com/2073-4395/12/10/2544](https://www.mdpi.com/2073-4395/12/10/2544)
33. Three-way PCA - Jorge N. Tendeiro, accessed April 17, 2025, [https://www.jorgetendeiro.com/talks/2010_JOCLAD_slides.pdf](https://www.jorgetendeiro.com/talks/2010_JOCLAD_slides.pdf)
34. A New Algorithm for Computing Disjoint Orthogonal Components in the Parallel Factor Analysis Model with Simulations and Applications to Real-World Data - MDPI, accessed April 17, 2025, [https://www.mdpi.com/2227-7390/9/17/2058](https://www.mdpi.com/2227-7390/9/17/2058)
35. (PDF) A New Algorithm for Computing Disjoint Orthogonal Components in the Parallel Factor Analysis Model with Simulations and Applications to Real-World Data - ResearchGate, accessed April 17, 2025, [https://www.researchgate.net/publication/355336422_A_new_algorithm_for_computing_disjoint_orthogonal_components_in_the_parallel_factor_analysis_model_with_simulations_and_applications_to_real-world_data](https://www.researchgate.net/publication/355336422_A_new_algorithm_for_computing_disjoint_orthogonal_components_in_the_parallel_factor_analysis_model_with_simulations_and_applications_to_real-world_data)
36. Degeneracy in Candecomp/Parafac and Indscal Explained For Several Three-Sliced Arrays With A Two-Valued Typical Rank, accessed April 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2806219/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2806219/)
37. Guaranteed Tensor Decomposition via Orthogonalized Alternating Least Squares - arXiv, accessed April 17, 2025, [https://arxiv.org/pdf/1703.01804](https://arxiv.org/pdf/1703.01804)
38. PARAFAC with orthogonality in one mode and applications in DS-CDMA systems | Request PDF - ResearchGate, accessed April 17, 2025, [https://www.researchgate.net/publication/224150287_PARAFAC_with_orthogonality_in_one_mode_and_applications_in_DS-CDMA_systems](https://www.researchgate.net/publication/224150287_PARAFAC_with_orthogonality_in_one_mode_and_applications_in_DS-CDMA_systems)
39. Partial uniqueness in CANDECOMP/PARAFAC - The Three-Mode Company, accessed April 17, 2025, [https://three-mode.leidenuniv.nl/pdf/t/tenberge2004a_joc.pdf](https://three-mode.leidenuniv.nl/pdf/t/tenberge2004a_joc.pdf)
40. On Uniqueness in Candecomp/Parafac | Psychometrika | Cambridge Core, accessed April 17, 2025, [https://www.cambridge.org/core/product/identifier/S003331230002500X/type/journal_article](https://www.cambridge.org/core/product/identifier/S003331230002500X/type/journal_article)
41. Kruskal's condition for uniqueness in Candecomp/Parafac when ranks and k-ranks coincide - Alwin Stegeman, accessed April 17, 2025, [http://www.alwinstegeman.nl/docs/Stegeman%20Ten%20Berge%20CSDA%20(2006).pdf](http://www.alwinstegeman.nl/docs/Stegeman%20Ten%20Berge%20CSDA%20(2006).pdf)
42. On uniqueness conditions for Candecomp/Parafac and Indscal with full column rank in one mode - Alwin Stegeman, accessed April 17, 2025, [http://www.alwinstegeman.nl/docs/Stegeman%20LAA%20(2009).pdf](http://www.alwinstegeman.nl/docs/Stegeman%20LAA%20(2009).pdf)
43. Kruskal's condition for uniqueness in Candecomp/Parafac when ranks and -ranks coincide | Request PDF - ResearchGate, accessed April 17, 2025, [https://www.researchgate.net/publication/4724140_Kruskal's_condition_for_uniqueness_in_CandecompParafac_when_ranks_and_-ranks_coincide](https://www.researchgate.net/publication/4724140_Kruskal's_condition_for_uniqueness_in_CandecompParafac_when_ranks_and_-ranks_coincide)
44. www.psychology.uwo.ca, accessed April 17, 2025, [https://www.psychology.uwo.ca/faculty/harshman/lawch5.pdf](https://www.psychology.uwo.ca/faculty/harshman/lawch5.pdf)
45. FOUNDATIONS OF THE PARAFAC PROCEDURE: MODELS AND CONDITIONS FOR AN "EXPLANATORY" MULTIMODAL FACTOR ANALYSIS - The Three-Mode Company, accessed April 17, 2025, [https://three-mode.leidenuniv.nl/pdf/h/harshman1970uclawpp.pdf](https://three-mode.leidenuniv.nl/pdf/h/harshman1970uclawpp.pdf)