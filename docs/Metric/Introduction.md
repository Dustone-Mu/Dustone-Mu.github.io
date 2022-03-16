---
sort: 1
---

# Introduction

The Metric module provides a series of widely used topic modeling metrics to evaluate these probabilistic topic models after training, including Accuracy for document classification, Word Perplexity for document modeling, Normalized Mutual Information for document clustering and Topic Coherence for measuring topic quality.

Metrics are as following:

|  Metrics                       |  API                                        |
|--------------------------------|---------------------------------------------|
|  Classification Accuracy       |  ACC(theta0, theta1, label0, label1, model) |
|  Cluster Accuracy              |  Cluster_ACC(y_true, y_pred)                |
|  Normalized Mutual Information |  NMI(y_true, y_pred)                        |
|  Perplexity                    |  Perplexity(x, x_reconstruct)               |
|  Poisson Likelihood            |  Poisson_Likelihood(x, x_re)                |
|  Reconstruction Error          |  Reconstruct_Error(x, x_re)                 |
|  Topic Coherence               |  Topic_Coherence                            |

All metric API are included in pydpm/_metric/...

> Example

The demo of Classification Accuracy: 

```python
from pydpm._metric import ACC
from pydpm._model import PGBN

# create the model and deploy it on gpu or cpu
model = PGBN([128, 64, 32], device='gpu')
model.initial(train_data)
train_local_params = model.train(100, train_data)
train_local_params = model.test(100, train_data)
test_local_params = model.test(100, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8549
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_label, test_label, 'SVM')

```
