#home

## Basic OLS

Begin with a [[A1|Linear Population Model]] i.e. $Y = X\beta + \varepsilon.$ Provided the [[design matrix]]  satisfies the [[A2|Full Rank Condition]] i.e. $\operatorname{Rank}(X)=K$ [[Result A|then]] $b^{\text{OLS}} = (X^T X)^{-1}X^T Y$ exists and is unique. 

### Interpreting OLS 


Via the [[Frisch-Waugh-Lovell Theorem]], we can show that for an individual coefficient:
$$b^{\text{OLS}}_j = (\mathbf{x}_j^T M_{-j} \mathbf{x}_j)^{-1} \mathbf{x}_j^T M_{-j} Y = (\mathbf{x}_j^{\star T} \mathbf{x}_j^{\star})^{-1} \mathbf{x}_j^{\star} Y$$
In particular, coefficients are [[partial regression coefficients]]: the effect of x_j on Y with the effect of X_-j having been netted, or [[partialed out]]. 

The [[FWL]] gives other useful results for interpretation of the OLS estimator. Firstly, it clearly [[Orthogonal Partitioned Regression|shows]] that if the explanatory variables are orthogonal then the slopes from estimating the model separately or together will coincide. 

We can [[projection interpretation of OLS|identify]] then that $Xb^{\text{OLS}}$ is exactly $P_x Y$, from which it follows our [[residuals]] are $M_x Y$.  In another view, the [[OLS normal equations]] are X^Te=0, and so the residuals are constructed to be orthogonal to the column space of X which gives the same decomposition. EDITOR NOTES: NEED TO LINK TO A RESULT THERE IG. 

FWL  [[implies]] that the regression of Y on X and an intercept is equivalent to the regression of the [[demeaned]] Y on the [[demeaned]] X.  As such in this case the OLS estimator (for the remaining slopes) is [[Interpreting OLS| exactly]]:

$b^{\text{OLS}} = (X^T X)^{-1}X^T Y= Varhat(X)^-1 Covhat(X,Y)


## The geometry of OLS and goodness-of-fit

An application of this decomposition is to observe the [[geometry of OLS]]:

$$\mathbf{y}^{\prime} \mathbf{y} = \mathbf{y}^{\prime} \mathbf{P}^{\prime} \mathbf{P y}+\mathbf{y}^{\prime} \mathbf{M}^{\prime} \mathbf{M y} = \hat{\mathbf{y}}^{\prime} \mathbf{y}+\mathbf{e}^{\prime} \mathbf{e}$$
EDITORS NOTE: replace the prime symbol with the ^T for transpose, and add a second line, the same, but with squared norms instead of inner products. also the second yhat is missing its hat.

If we are working with the [[demeaned]] regression then this is exactly the [[ANOVA decomposition]] in line 2.

Otherwise, define the demeaning matrix M^0. If X contains an intercept then the first normal equation is 1^Te= sum to n e=0 which further implies Ybar=betahatXbar.  These results are sufficient for:

y M0 y = b X M0 Xb + e e = expression with sum notation = SST = SSE+SSR

The usual [[coefficient of determination]] then follows:  R^2=SSR/ SST = b X M0Xb y M0y = 1 − e e y M0y , which can be [[Coefficient of Determination as Squared Correlation]] to be the squared correlation between the observed values of yi and the predictions yihat.

In the case that Y and X are random variables then the ANOVA is clearly the finite sample analogue of the [[Law of total variance]], and the R^2 is the fraction of Y's variance reduced by expanding the information set to include X. That is,

1/n SST = 1/n SSR + 1/n SSE --> Varhat(Y)= Varhat(yhat) +varhat(epsilon)

where yhat is our regression function E(Y|X) by assumption.

[[NOTE HERE 1THIS COULD BE CLEANER BUT THE EXACT RELATIONSHIP IS UNCLEAR TO ME]]












## Basic Properties of the OLS estimator 

If we further specify [[C1|Moment Orthogonality]] i.e. $E(\mathbf{x}_i \varepsilon_i) = \mathbf{0}_K$, and some [[Assumptions for basic Asymptotics|Asymptotic's Assumptions]]--  ${(\varepsilon_i\mathbf{x}_i)}$ being i.i.d with finite variance is strongly sufficient-- then we have a [[consistent]] estimator i.e. $b^{\text{OLS}} \xrightarrow{p} \beta$ and $b^{\text{OLS}}$ is [[Result C2|asymptotically normal]].

If we strengthen [[C1|Moment Orthogonality]] to pure [[B|Exogeneity]] i.e.  $E(\varepsilon \mid X)=\mathbf{0}_n.$ then we also have $E(b^{\text{OLS}})=\beta$ i.e. [[Result B|Finite Sample Unbiasedness]].

Provided the [[disturbances]] are minimally well-behaved i.e. $\operatorname{Var}(\varepsilon \mid X)=\Sigma$ we can[[Result E| show]]:
$$\mathbb{V}\text{ar}(b^{\text{OLS}} \mid X) = (X^T X)^{-1} X^T \Sigma X (X^T X)^{-1} = \frac{1}{n^2} S_{xx}^{-1} X^T \Sigma X S_{xx}^{-1}$$
If, in particular, we assume [[spherical disturbances]] i.e. $\mathbb{V}\text{ar}(\epsilon|X) = \sigma^2 I$, then [[Result E2| clearly]]:
$$\mathbb{V}\text{ar}(b^{\text{OLS}} \mid X) =\sigma^2 (X^T X)^{-1} $$
The natural (unbiased) estimator of $\sigma^2$ is $s^2 = \frac{\mathbf{e}'\mathbf{e}}{n - K}$; substituting this into the above gives the estimated sampling variance. In particular, the standard error of $b_k$ is

$$\text{SE}(b_k) = \sqrt{s^2 (X'X)^{-1}_{kk}} = \sqrt{\widehat{\mathbb{V}\text{ar}}(b_k^{\text{OLS}} \mid X)}$$

Importantly, under this particular [[stochastic specification]], the OLS estimator is minimum variance linear unbiased estimator by [[Gauss-Markov Theorem]].  The component wise standard errors of any other estimator are greater than or equal to expression above.

For small-sample inference we will need a stronger (distributional) assumption, in particular we will take the [[disturbances]] to be jointly normal. Then $b_k \sim N\left[\beta_k, \sigma^2 S^{kk}\right]$, and;

$$z_k = \frac{b_k - \beta_k}{\sqrt{\sigma^2 S^{kk}}} \sim N(0,1)$$

Since of course $\sigma^2$ is not known we use the usual:

$$t_k = \frac{b_k - \beta_k}{\sqrt{s^2 S^{kk}}} \sim t_{n-K}$$

In any case, this test statistic is asymptotically standard normal provided the conditions outlined above (not including exogeneity or normal errors) hold.  For testing the significance of a subset of coefficients we make use of the [[F-test]] which in finite samples requires again that $b_k \sim N\left[\beta_k, \sigma^2 S^{kk}\right]$. Importantly, whether X is fixed or random or a combination of both, these tests are valid asymptotically and finately under the provided assumptions.

A question that is usually of interest is whether the regression equation as a whole is significant. This test is a joint test of the hypotheses that all the coefficients except the constant term are zero. If all the slopes are zero, then the multiple correlation coefficient is zero as well, so we can base a test of this hypothesis on the value of R2. The central result needed to carry out the test is the distribution of the statistic F [K − 1, n − K] = R2/(K − 1) (1 − R2)/(n − K) . (4-15) If the hypothesis that β2 = 0 (the part of β not including the constant) is true and the disturbances are normally distributed, then this statistic has an F distribution with K−1 and n − K degrees of freedom.3 Large values of F give evidence against the validity of the hypothesis. Note that a large F is induced by a large value of R2. The logic of the test is that the F statistic is a measure of the loss of fit (namely, all of R2) that results when we impose the restriction that all the slopes are zero. If F is large, then the hypothesis is rejected

## Model Specification




