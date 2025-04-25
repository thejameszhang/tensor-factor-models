### Key Points
- Research suggests that orthogonality constraints in PARAFAC decompositions can improve stability and interpretability but may reduce fit to data.
- It seems likely that additional research ideas from related papers include exploring frequency-domain analysis and developing new multi-horizon pricing models.
- The evidence leans toward significant pricing errors in characteristic-sorted portfolios over extended horizons, suggesting areas for theoretical and empirical advancement.

---

### Direct Answer

Here’s a clear and simple breakdown of the additional research ideas from the other papers and the pros, cons, and theoretical consequences of using orthogonality constraints in PARAFAC decompositions, based on the information available.

#### Additional Research Ideas
The other papers provide several ideas that could build on your paper’s focus on the term structure of expected returns using PARAFAC decomposition. These include:
- **Frequency-Domain Analysis**: Look into how returns behave at different frequencies, like using moving averages or bandpass filters to understand risk over various time horizons.
- **New Pricing Models**: Develop models that better handle returns over multiple time periods, especially for long-term investments like bonds.
- **Portfolio Strategies**: Explore how to build portfolios that account for past and recent firm characteristics to improve efficiency and reduce risks.
- **Economic Drivers**: Investigate why returns decay differently for stocks with varying market risks and what economic factors might explain this.
- **Testing and Robustness**: Test these models in real-world scenarios and check their reliability across different markets or longer time frames.

These ideas can help refine your approach to understanding how expected returns change over time and across different stocks.

#### Orthogonality Constraints in PARAFAC
Orthogonality constraints mean making the factors in your PARAFAC model uncorrelated, which can have both benefits and drawbacks:
- **Pros**: It can make your model more stable and easier to interpret, especially for exploratory studies. It also helps in breaking down variance into clear parts, which is useful in certain analyses.
- **Cons**: It might not fit the data as well, especially in finance where factors (like time, stocks, and lags) are often related. It can also make interpretation harder in some cases and add complexity.
- **Theoretical Consequences**: If factors aren’t naturally uncorrelated, forcing them to be could misrepresent the data, affecting how we understand the model. It might also limit the model’s flexibility, which is important in complex financial systems.

Given your paper’s focus on finance, it seems like avoiding these constraints might be better to keep the model flexible, but exploring them could be useful in specific scenarios where factors are known to be uncorrelated.

---

---

### Survey Note: Detailed Analysis of Research Ideas and Orthogonality Constraints in PARAFAC Decompositions

This note provides a comprehensive analysis of additional research ideas derived from the provided attachments and a detailed examination of the pros, cons, and theoretical consequences of using orthogonality constraints in PARAFAC decompositions, as requested. The analysis is grounded in the context of the paper "Term Structure of Firm Characteristics and Multi-Horizon Investment" (TSFC-paper.pdf), which leverages PARAFAC decomposition to study the term structure of expected returns across multiple time horizons using a tensor factor model.

#### Research Ideas from Other Papers

The attachments include several papers and discussions that offer complementary research directions, particularly in asset pricing, multi-horizon returns, and characteristic-based portfolios. Below, we summarize the main ideas and potential extensions, organized by source.

##### Stefano Giglio's Discussion - Measuring Horizon-Specific Systematic Risk via Spectral Betas (id:7)
This discussion, dated September 27, 2018, reviews a paper by Bandi, Chaudhuri, Lo, and Tamoni on measuring horizon-specific systematic risk using spectral betas. The paper decomposes time series of returns into orthogonal components using moving averages to analyze covariance at different frequencies, finding that low-frequency (long-horizon) covariance better explains expected returns. Key research ideas include:
- **Frequency-Domain Analysis in Asset Pricing**: Further theoretical and empirical work is needed to understand frequency-domain representations of asset pricing models. This could involve decomposing returns into components at different frequencies to assess their impact on risk premia, aligning with the TSFC paper’s focus on multi-horizon returns.
- **Linking Decomposition to Specific Models**: Explore how such decompositions can be integrated with specific asset pricing models, such as the Consumption Capital Asset Pricing Model (CCAPM), where long-run averages (e.g., Parker and Julliard, 2005) are used to capture slow consumption responses.
- **Restrictions to Trade at Different Frequencies**: Investigate how restrictions on trading at different frequencies, as explored by Crouzet, Dew-Becker, and Nathanson (2018), might affect asset pricing, particularly in relation to horizon-specific risk.
- **Alternative Filtering Techniques**: Compare the moving averages used in the paper with standard bandpass filters to see if empirical results change, potentially leading to more robust methods for analyzing horizon-specific risk.
- **New Tools for Dynamics**: Develop new methodologies to better understand the dynamics of asset pricing models in the frequency domain, described as a "very interesting research agenda" in the discussion.

These ideas could extend the TSFC paper by incorporating frequency-domain techniques to enhance the analysis of term structure dynamics.

##### Kent Daniel's Discussion - New and Old Sorts (id:9)
This discussion, presented at the 2021 AFA Meetings on January 3, 2021, comments on Baba Yara, Boons, and Tamoni’s paper (id:11) on characteristic-sorted portfolios. It highlights findings that older sorts (portfolios based on lagged characteristics) often show negative alphas, suggesting returns decay faster than characteristic spreads. Research ideas include:
- **Exploring Hypotheses on Return Decay**: Investigate why older sorts have negative alphas and why returns decay faster than characteristic spreads. This could involve testing hypotheses about market dynamics, behavioral factors, or structural economic changes.
- **Improving Portfolio Efficiency**: Explore how combining characteristic-based portfolios with hedge portfolios can reduce unpriced risk while preserving expected returns, building on works like Daniel, Mota, Hottke, and Santos (2020).
- **Dynamics of Idiosyncratic Volatility**: Study the high autocorrelation of idiosyncratic volatility over long periods (e.g., Rachwalski and Wen, 2016), which shows significant cross-sectional correlations even years after portfolio formation, and how it can be modeled in asset pricing.
- **Addressing Model Misspecification**: Develop more accurate characteristic models that account for changes in characteristics over time, potentially integrating dynamic factors or time-varying risk premiums, given the suggestion that current models are misspecified.

These ideas align with the TSFC paper’s focus on lagged characteristics and could enhance its analysis of return dynamics over extended horizons.

##### Chernov, Lochstoer, Lundeby (2019) - Conditional Dynamics and the Multi-Horizon Risk-Return Trade-Off (id:8)
This paper proposes using multi-horizon returns (MHR) as endogenous test assets to evaluate asset-pricing models, finding that models like CAPM and Fama-French struggle to price longer-horizon returns due to misspecified conditional dynamics. Research ideas include:
- **Improving Estimation of Conditional Dynamics**: Focus on developing econometric or machine learning techniques to better estimate the conditional mean and variance processes of factors, addressing the "Herculean task" mentioned in the paper.
- **Understanding Economic Drivers**: Investigate economic forces (e.g., behavioral factors, market frictions) behind conditional dynamics, given the positive relationship between model complexity and pricing errors.
- **Developing New Models for Multi-Horizon Pricing**: Create theoretical frameworks or modify existing models (e.g., incorporating stochastic volatility, regime-switching) to better capture multi-horizon pricing dynamics, crucial for applications like capital budgeting.
- **Expanding Test Assets**: Test models using a broader range of assets (e.g., portfolios, individual securities) to uncover additional conditional misspecifications, enhancing the robustness of findings.
- **Out-of-Sample Testing**: Explore real-time, out-of-sample testing strategies for factor timing, building on approaches like Moreira and Muir (2017) and Haddad, Kozak, and Santosh (2020), to ensure practical applicability.
- **Connection to Long-Horizon Assets**: Apply MHR testing to long-horizon asset classes like bonds or dividend strips to assess whether similar misspecifications exist, connecting to literature on zero-coupon assets.
- **Statistical Power and Robustness**: Explore the robustness of the MHR-based Generalized Method of Moments (GMM) test under different market conditions, sample sizes, or model specifications to ensure reliability.

These ideas directly complement the TSFC paper’s multi-horizon focus, offering avenues to improve model accuracy and empirical testing.

##### Baba Yara, Boons, Tamoni (2020) - New and Old Sorts: Implications for Asset Pricing (id:11)
This paper examines returns of characteristic-sorted portfolios over extended horizons, finding significant pricing errors when comparing new sorts (recent characteristics) and old sorts (past characteristics). Research ideas include:
- **Parsimonious Factor Representation**: Develop a simpler factor model that can jointly price new and old sorts, addressing the challenge of finding the most parsimonious representation without overfitting, as suggested for future work.
- **Understanding Economic Drivers of Pricing Errors**: Investigate the economic mechanisms behind pricing errors, particularly their link to market beta, and why returns decay at different rates for characteristics with varying betas.
- **Joint Pricing Challenges**: Further investigate the difficulty of jointly pricing new and old sorts, potentially through hybrid models or refined econometric techniques, given the trade-off observed between smaller and larger models.
- **Portfolio Construction Implications**: Explore practical portfolio strategies that account for the dynamics of firm characteristics over time, balancing transaction costs (e.g., less rebalancing for old stock portfolios) and risk-adjusted returns.
- **Extending Horizon and Scope**: Analyze pricing errors over longer horizons (e.g., 10-15 years, as in Keloharju et al., 2019; Cho and Polk, 2019) or across different markets to test universality, enhancing the scope of findings.
- **Theoretical Model Development**: Build theoretical models that incorporate past characteristics and their changes, potentially integrating firm entry/exit, market sentiment, or liquidity constraints, challenging existing models like Gomes et al. (2003) and Zhang (2005).

These ideas provide a rich agenda for extending the TSFC paper, particularly in refining factor models and understanding long-term return dynamics.

#### Orthogonality Constraints in PARAFAC Decompositions

PARAFAC decomposition, as used in the TSFC paper, involves decomposing a three-dimensional tensor (time, stocks, lags) into a sum of rank-1 tensors without necessarily imposing orthogonality constraints. Orthogonality constraints would require the factor matrices across different modes to be uncorrelated, which has both benefits and drawbacks, as detailed below.

##### Pros of Orthogonality Constraints
- **Improved Stability**: Orthogonality can stabilize PARAFAC solutions, particularly in psychometrics, where unstable solutions are a concern due to noisy or ill-conditioned data. This is noted in the attachment "PARAFAC.pdf" (id:13), under section 4.3, as a means to overcome problems with unstable solutions.
- **Enhanced Interpretability**: For exploratory analyses, especially with experimentally designed studies, orthogonality allows the sum-of-squares (variance) to be partitioned into contributions from individual components, making it easier to understand each factor’s role. This is particularly valuable in fields like psychometrics, as mentioned in "PARAFAC.pdf."
- **Mathematical Uniqueness**: PARAFAC decompositions remain unique even with orthogonality constraints, preserving a key property (least squares solutions under constraints), as discussed in "PARAFAC.pdf."
- **Facilitates Variance Partitioning**: Orthogonality enables isolating and analyzing the variance explained by each component, useful in experimental designs, as noted in "PARAFAC.pdf."

##### Cons of Orthogonality Constraints
- **Reduced Fit to Data**: Imposing orthogonality always results in a lower fit compared to unconstrained PARAFAC, as it restricts the model’s flexibility. This is explicitly stated in "PARAFAC.pdf," noting that constrained models have lower fit than unconstrained ones.
- **Hindered Interpretation in Some Fields**: In chemometrics, orthogonality is less common because it can obscure direct relationships between loadings and physical properties (e.g., spectra of analytes), making interpretation less intuitive, as per "PARAFAC.pdf."
- **Increased Complexity**: Adding orthogonality constraints introduces additional optimization challenges, potentially leading to overfitting if the data does not align with the imposed structure, as inferred from the complexity discussion in "PARAFAC.pdf."
- **Limited Applicability**: Orthogonality is more common in psychometrics than in chemometrics or finance, where other constraints (e.g., nonnegativity) may be preferred, as noted in "PARAFAC.pdf," limiting its general applicability.

##### Theoretical Consequences
- **Uniqueness and Identifiability**: Orthogonality constraints can affect the uniqueness of PARAFAC decompositions. While PARAFAC is generally unique under conditions like Kruskal’s condition, imposing orthogonality might alter these conditions or lead to different solutions, as suggested by the mathematical uniqueness discussion in "PARAFAC.pdf."
- **Model Misspecification**: If the true underlying factors are not orthogonal, imposing orthogonality can lead to model misspecification, where the decomposed factors do not accurately reflect the data-generating process. This is particularly relevant in finance, where factors across time, stocks, and lags may be interrelated, as inferred from the TSFC paper’s context.
- **Interpretation in Asset Pricing**: In the context of the TSFC paper, orthogonality might imply that factors across different modes are uncorrelated, which may not align with economic theory. For example, stock returns across different lags might naturally be correlated due to market dynamics or firm-specific characteristics, potentially distorting the model’s economic interpretation.
- **Flexibility vs. Parsimony**: Orthogonality reduces the model’s flexibility, which can be a drawback in complex systems like financial markets, where interactions between modes are likely to be non-orthogonal, as suggested by the need for flexibility in "PARAFAC.pdf."

##### Additional Insights from Literature
A web search on "orthogonality constraints in PARAFAC decompositions" revealed additional context, particularly from [Overview of constrained PARAFAC models](https://asp-eurasipjournals.springeropen.com/articles/10.1186/1687-6180-2014-142), which discusses constrained PARAFAC models with linear dependencies, and [Parafac with orthogonality in one mode and applications in DS-CDMA systems](https://ieeexplore.ieee.org/document/5495717/), which shows improved performance when signals are uncorrelated, suggesting orthogonality is beneficial in specific contexts like signal processing.

Given the TSFC paper’s focus on finance, where interpretability is less straightforward than in chemometrics and factors may not be naturally orthogonal, it seems likely that the paper avoided orthogonality constraints to preserve the best fit to the data. However, exploring these constraints could be valuable in future research, especially if factors are known to be uncorrelated, enhancing stability and interpretability in specific scenarios.

#### Summary Table: Research Ideas and Orthogonality Constraints

To organize the information, below is a table summarizing the key research ideas and the pros/cons of orthogonality constraints:

| **Source**                          | **Key Research Ideas**                                                                 |
|-------------------------------------|---------------------------------------------------------------------------------------|
| Stefano Giglio's Discussion (id:7)  | Frequency-domain analysis, linking to models, trade restrictions, alternative filters, new tools for dynamics. |
| Kent Daniel's Discussion (id:9)     | Explore return decay, improve portfolio efficiency, study idiosyncratic volatility, address model misspecification. |
| Chernov et al. (2019) (id:8)        | Improve conditional dynamics estimation, understand economic drivers, new multi-horizon models, expand test assets, out-of-sample testing, long-horizon assets, robustness. |
| Baba Yara et al. (2020) (id:11)     | Parsimonious factor representation, economic drivers of errors, joint pricing, portfolio strategies, extend horizons, theoretical models. |

| **Aspect**                  | **Pros of Orthogonality**                              | **Cons of Orthogonality**                              |
|-----------------------------|-------------------------------------------------------|-------------------------------------------------------|
| Stability                   | Improves stability, especially in noisy data.          | May increase complexity, risking overfitting.          |
| Interpretability            | Enhances for exploratory studies, partitions variance. | Hinders in chemometrics, obscures physical relationships. |
| Fit to Data                 | -                                                     | Reduces fit compared to unconstrained models.          |
| Applicability               | Common in psychometrics.                               | Limited in finance, prefers other constraints.         |

This table encapsulates the key findings and provides a structured overview for further exploration.

#### Conclusion

The research ideas from the other papers offer a robust foundation for extending the TSFC paper, particularly in refining multi-horizon models, exploring frequency-domain techniques, and addressing pricing errors in characteristic-sorted portfolios. Regarding orthogonality constraints in PARAFAC, while they offer benefits in stability and interpretability, their drawbacks in fit and applicability suggest caution, especially in finance, where flexibility is crucial. Future work could explore these constraints in specific contexts where factors are uncorrelated, balancing interpretability and fit.

---

### Key Citations
- [Overview of constrained PARAFAC models EURASIP Journal](https://asp-eurasipjournals.springeropen.com/articles/10.1186/1687-6180-2014-142)
- [Parafac with orthogonality in one mode IEEE Conference](https://ieeexplore.ieee.org/document/5495717/)