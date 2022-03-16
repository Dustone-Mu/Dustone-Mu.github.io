---
sort: 1
---

# Introduction

## Sampler Module

The Sampler module contains distribution and model sampler parts, both of which can be deployed on the platform with either CPU or GPU. Specifically, the former part, distribution sampler, has contained all of the basic distributions included in numpy (Van Der Walt et al., 2011), tensorflow (Abadi et al., 2016) and torch (Paszke et al., 2019), and the latter part, model sampler, provides additional model-specific samplers for updating the parameters in probabilistic topic models. Note that we have also provided both CPU and GPU versions for the Sampler module. 

Specifically, the mainly difference between our Sampler module and the widely used numpy.random module is that you need to instantiate a sampler object and deploy it on CPU or GPU before starting sampling. Moreover, our Sampler module provides an interface parameter to indicate how many repetitions of distribution sampling are required, which makes our Sampler module more flexible than numpy.random in sampling some complicated distributions, like Multinomial and Dirchilet distributions. The sampling efficiency comparisons between our Sampler module and other libraries have been exhibited in [*Performance Evaluation*](https://dustone-mu.github.io/Sample/Performance%20Evaluation.html) part.

## About GPU Acceleration

The Graphics Processing Unit (GPU) provides much higher instruction throughput and memory bandwidth than the CPU within a similar price and power envelope. Many applications leverage these higher capabilities to run faster on the GPU than on the CPU.

In general, an application has a mix of parallel parts and sequential parts, so systems are designed with a mix of GPUs and CPUs in order to maximize overall performance. Applications with a high degree of parallelism can exploit this massively parallel nature of the GPU to achieve higher performance than on the CPU.

So pydpm take advantage of GPU parallelism by programme the probability distribution functions with CUDA C++ to accelerate the speed of sampler.


## Distribution Function List

The parameters of partial distribution functions are as following:

|Function        | Parameters List   | 
|----------------|-------------------|
|Normal          |mean, std, times   |
|Multinomial     |count, prob, times |
|Poisson         |lambda, times      |
|Gamma           |shape, scale, times|
|Beta            |alpha, beta, times |
|F               |n1, n2, times      |
|StudentT        |n, times           |
|Dirichlet       |alpha, times       |
|Crt             |point, p, times    |
|Weibull         |shape, scale, times|
|Chisquare       |n, times           |
|Geometric       |p, times           |
|...             |...                |

All distribution fuctions can be found in [*Distribution Sampler*](https://dustone-mu.github.io/Sample/Distribution%20Sampler.html) part.


## Example

The sampler demo of gamma distribution is as following:

```python
from pydpm._sampler import Basic_Sampler

sampler = Basic_Sampler('gpu')
a = sampler.gamma(np.ones(100)*5, 1, times=10)
b = sampler.gamma(np.ones([100, 100])*5, 1, times=10)
```

More sampler demos can be found in pydpm/_sampler/... and [*Distribution Sampler*](https://dustone-mu.github.io/Sample/Distribution%20Sampler.html) part.







