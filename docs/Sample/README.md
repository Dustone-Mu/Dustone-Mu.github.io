---
sort: 2
---

# Sample

{% include list.liquid %}


>Sample on GPU

The Graphics Processing Unit (GPU) provides much higher instruction throughput and memory bandwidth than the CPU within a similar price and power envelope. Many applications leverage these higher capabilities to run faster on the GPU than on the CPU.

In general, an application has a mix of parallel parts and sequential parts, so systems are designed with a mix of GPUs and CPUs in order to maximize overall performance. Applications with a high degree of parallelism can exploit this massively parallel nature of the GPU to achieve higher performance than on the CPU.

So pydpm take advantage of GPU parallelism by programme the probability distribution functions with CUDA C++ to accelerate the speed of sampler.


>Function list

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

>Example

```python
from pydpm._sampler import Basic_Sampler

sampler = Basic_Sampler('gpu')
a = sampler.gamma(np.ones(100)*5, 1, times=10)
b = sampler.gamma(np.ones([100, 100])*5, 1, times=10)
```
More sampler demos can be found in pydpm/_sampler/...


> Sample

In order to verify the accuracy of the sampler function, comparsions between the distribution of the sampling results and the actual distribution function are implemented. The results are as following

> gamma distribution

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_gamma.png)

> multinomial distribution

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_multinomial.png)


The integral comparsion of sampler results and standard distribution can be found in pydpm/_example/Sampler_Demo.py

>Compare
>
Compare the sampling speed of distribution functions with numpy:
![Image text](https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_numpy.png)  
The compared code can be found in pydpm/example/Sampler_Speed_Demo.py

Compare the sampling speed of distribution functions with tensorflow and torch:
![Image text](https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_tf2_torch.png)  
The compared code can be found in pydpm/example/Sampler_Speed_Demo.py

