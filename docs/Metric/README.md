---
sort: 3
---

# Metric

{% include list.liquid %}

Pydpm provides a variety of metric functions for evaluation the performance of topic models.

Metrics are as following:

|Metrics                                                 |  API       |
|------------------------------------------|----------|
|  Classification Accuracy                       |      ACC(theta0, theta1, label0, label1, model)       |
|  Cluster Accuracy                                |     Cluster_ACC(y_true, y_pred) |
|  Normalized Mutual Information        |     NMI(y_true, y_pred)             |
|  Perplexity                                          |  Perplexity(x, x_reconstruct)     |
|  Likelihood                                         |  Poisson_Likelihood(x, x_re)      |
|  Reconstruction                                  |  Reconstruct_Error(x, x_re)       |
|  Coherence                                        |  Topic_Coherence                    |

All metric API are included in pydpm/_metric/...

> Example

```python
acc = pydpm._metric.ACC(tr_theta, te_theta, tr_label, te_label, 'SVM')
```