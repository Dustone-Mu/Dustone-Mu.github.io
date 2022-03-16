---
sort: 2
---

# Metrics

> Metrics List

|  Metrics                       |  API                                 |
|--------------------------------|--------------------------------------|
|  Classification Accuracy       |  ACC(x_tr, x_te, y_tr, y_te, model)  |
|  Cluster Accuracy              |  Cluster_ACC(y_true, y_pred)         |
|  Normalized Mutual Information |  NMI(y_true, y_pred)                 |
|  Perplexity                    |  Perplexity(x, x_reconstruct)        |
|  Poisson Likelihood            |  Poisson_Likelihood(x, x_re)         |
|  Reconstruction Error          |  Reconstruct_Error(x, x_re)          |
|  Topic Coherence               |  Topic_Coherence                     |

All metric API are included in pydpm/_metric/...



## Classification Accuracy

> API

pydpm._metirc.ACC(x_tr, x_te, y_tr, y_te, model='SVM')

> Detail

```python
classifier = sklearn.svm.SVC()
classifier.fit(x_tr, y_tr)
acc = classifier.score(x_te, y_te)
```



## Cluster Accuracy

> API

pydpm._metirc.Cluster_ACC(y_true, y_pred)

> Detail

```math
AC = (\sum_{i=1}^n \sigma(y_{true}^i, map(y_{pred}^i))/N
```

$\sigma$ is indicator function:

$$
\sigma(x, y) = 
\begin{cases}
1, &if x=y\\
0, &otherwise
\end{cases}
$$



## Normalized Mutual Information

> API

pydpm._metirc.NMI(y_true, y_pred)

> Detail

```math
lin = sum(X * logmax(X_re) - X_re - logmax(gamma(X_re + 1)))
```



## Perplexity

> API

pydpm._metirc.Poisson_Linkelihood(X, X_re)

> Detail

```math
sum(X * logmax(X_re) - X_re - logmax(gamma(X_re + 1)))
```



## Poisson Likelihood

> API

pydpm._metirc.Poisson_Likelihood(X, X_re)

> Detail

```math
likelihood = \sum_i(X^i * log(X_{re}^i) - X_{re}^i - log(\gamma(X_{re}^i + 1)))
```



## Reconstruction Error

> API

pydpm._metirc.Poisson_Linkelihood(X, X_re)

> Detail

```math
e = \sum_i((X^i - X_{re}^i)^2)
```




## Topic Coherence

> API

pydpm._metirc.Poisson_Linkelihood(X, X_re)

> Detail

```math
sum(X * logmax(X_re) - X_re - logmax(gamma(X_re + 1)))
```




