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
|  Perplexity                    |  Perplexity(x, x_re)                 |
|  Poisson Likelihood            |  Poisson_Likelihood(x, x_re)         |
|  Reconstruction Error          |  Reconstruct_Error(x, x_re)          |
|  Topic Coherence               |  Topic_Coherence                     |

All metric API are included in pydpm/_metric/...



## Classification Accuracy

> API

```python
pydpm._metirc.ACC(x_tr, x_te, y_tr, y_te, model='SVM')
'''
Inputs:
    x_tr : [np.ndarray] K*N_train matrix, N_train latent features of length K
    x_te : [np.ndarray] K*N_test matrix, N_test latent features of length K
    y_tr : [np.ndarray] N_train vector, labels of N_train latent features
    y_te : [np.ndarray] N_test vector, labels of N_test latent features

Outputs:
    accuracy: [float] scalar, the accuracy score
'''
```

> Detail

```python
classifier = sklearn.svm.SVC()
classifier.fit(x_tr, y_tr)
acc = classifier.score(x_te, y_te)
```



## Cluster Accuracy

> API

```python
pydpm._metirc.Cluster_ACC(y_true, y_pred)
'''
Inputs:
    y: the ground_true, shape:(n_sample,)
       ypred: pred_label, shape:(n_sample,)

Outputs:
    accuracy of cluster, in [0, 1]
'''
```

> Detail

$$
AC = \frac {\sum_{i=1}^n \sigma(y_{true}^i, map(y_{pred}^i))} {N}
$$

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

```python
pydpm._metirc.NMI(y_true, y_pred)
'''
Inputs:
    A: [int], ground_truth, shape:(n_sample,)
    B: [int], pred_label, shape:(n_sample,)

Outputs:
    NMI: [float], Normalized Mutual information of A and B
'''
```

> Detail

$$
NMI(X,Y) = \frac {2MI(X, Y)} {H(X)+H(Y)}\\
MI(X,Y) = \sum_{i=1}^{|X|} \sum_{j=1}{|Y|} P(i,j) log(\frac {P(i,j)} {P(i)P\`(j)})\\
H(X) = -\sum_{i=1}^{|X|} P(i)log(P(i))\\
H(Y) = -\sum_{j=1}^{|Y|} P\`(j)log(P\`(j))
$$



## Perplexity

> API

```python
pydpm._metirc.Perplexity(x, x_re)
'''
Inputs:
	x: [float] np.ndarray, V*N_test matrix, the observations for test_data
	x_hat: [float] np.ndarray, V*N_reconstruct matrix
Outputs:
	PPL: [float], the perplexity score
'''
```

> Detail

$$
PPL = exp(- \frac {\sum log(p(w))} {\sum_{d=1}^{M} N_d})
$$




## Poisson Likelihood

> API

```python
pydpm._metirc.Poisson_Likelihood(X, X_re)
```

> Detail

$$
likelihood = \sum_i(X^i * log(X_{re}^i) - X_{re}^i - log(\gamma(X_{re}^i + 1)))
$$



## Reconstruction Error

> API

```python
pydpm._metirc.Reconstruct_Error(x, x_re)
```

> Detail

$$
e = \sum (xi - x)^2
$$



## Topic Coherence

> API

```python
pydpm._metirc.Topic_Coherence(X, X_re)
```

> Detail

```python
gensim.models.coherencemodel.CoherenceModel().get_coherence()
```


