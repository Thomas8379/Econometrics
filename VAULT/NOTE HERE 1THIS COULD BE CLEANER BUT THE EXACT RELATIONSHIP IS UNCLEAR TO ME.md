- From Hayeshi
- 
- Uncentered $\boldsymbol{R}^{\mathbf{2}}$. One measure of the variability of the dependent variable is the sum of squares, $\sum y_i^2=\mathbf{y}^{\prime} \mathbf{y}$. Because the OLS residual is chosen to satisfy the normal equations, we have the following decomposition of $\mathbf{y}^{\prime} \mathbf{y}$ :

$$
\begin{aligned}
\mathbf{y}^{\prime} \mathbf{y} & =(\hat{\mathbf{y}}+\mathbf{e})^{\prime}(\hat{\mathbf{y}}+\mathbf{e}) \quad(\text { since } \mathbf{e}=\mathbf{y}-\hat{\mathbf{y}}) \\
& =\hat{\mathbf{y}}^{\prime} \hat{\mathbf{y}}+2 \hat{\mathbf{y}}^{\prime} \mathbf{e}+\mathbf{e}^{\prime} \mathbf{e} \\
& =\hat{\mathbf{y}}^{\prime} \hat{\mathbf{y}}+2 \mathbf{b}^{\prime} \mathbf{X}^{\prime} \mathbf{e}+\mathbf{e}^{\prime} \mathbf{e} \quad(\text { since } \hat{\mathbf{y}} \equiv \mathbf{X b}) \\
& =\hat{\mathbf{y}}^{\prime} \hat{\mathbf{y}}+\mathbf{e}^{\prime} \mathbf{e} \quad\left(\text { since } \mathbf{X}^{\prime} \mathbf{e}=\mathbf{0} \text { by the normal equations; see }\left(1.2 .3^{\prime}\right)\right)
\end{aligned}
$$


The uncentered $\boldsymbol{R}^{\mathbf{2}}$ is defined as

$$
R_{u c}^2 \equiv 1-\frac{\mathbf{e}^{\prime} \mathbf{e}}{\mathbf{y}^{\prime} \mathbf{y}}
$$


Because of the decomposition (1.2.15), this equals

$$
\frac{\hat{\mathbf{y}}^{\prime} \hat{\mathbf{y}}}{\mathbf{y}^{\prime} \mathbf{y}} .
$$


Since both $\hat{\mathbf{y}}^{\prime} \hat{\mathbf{y}}$ and $\mathbf{e}^{\prime} \mathbf{e}$ are nonnegative, $0 \leq R_{u c}^2 \leq 1$. Thus, the uncentered $R^2$ has the interpretation of the fraction of the variation of the dependent variable that is attributable to the variation in the explanatory variables. The closer the fitted value tracks the dependent variable, the closer is the uncentered $R^2$ to one.
- (Centered) $\boldsymbol{R}^{\mathbf{2}}$, the coefficient of determination. If the only regressor is a constant (so that $K=1$ and $x_{i 1}=1$ ), then it is easy to see from (1.2.5) that $\mathbf{b}$ equals $\bar{y}$, the sample mean of the dependent variable, which means that $\hat{y}_i=\bar{y}$ for all $i, \hat{\mathbf{y}}^{\prime} \hat{\mathbf{y}}$ in (1.2.15) equals $n \bar{y}^2$, and $\mathbf{e}^{\prime} \mathbf{e}$ equals $\sum_i\left(y_i-\bar{y}\right)^2$. If the regressors also include nonconstant variables, then it can be shown (the proof is left as an analytical exercise) that $\sum_i\left(y_i-\bar{y}\right)^2$ is decomposed as

$$
\sum_{i=1}^n\left(y_i-\bar{y}\right)^2=\sum_{i=1}^n\left(\hat{y}_i-\bar{y}\right)^2+\sum_{i=1}^n e_i^2 \text { with } \bar{y} \equiv \frac{1}{n} \sum_{i=1}^n y_i .
$$


The coefficient of determination, $R^2$, is defined as

$$
R^2 \equiv 1-\frac{\sum_{i=1}^n e_i^2}{\sum_{i=1}^n\left(y_i-\bar{y}\right)^2}
$$