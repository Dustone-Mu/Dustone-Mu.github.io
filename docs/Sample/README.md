---
sort: 2
---

# Sample

{% include list.liquid %}



Sample on GPU
=============
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

>Compare
>
Compare the sampling speed of distribution functions with numpy:
![Image text](https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_numpy.png)  
The compared code can be found in pydpm/example/Sampler_Speed_Demo.py

Compare the sampling speed of distribution functions with tensorflow and torch:
![Image text](https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_tf2_torch.png)  
The compared code can be found in pydpm/example/Sampler_Speed_Demo.py
