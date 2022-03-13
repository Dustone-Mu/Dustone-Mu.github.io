---
sort: 3
---

# Distribution Sampelr

## Introduction

In this part, comparsion between actual sampler results and standard distribution is implemented, and all distribution sampler demo are exhibited as well.

Code can be found in pydpm/example/Sampler_Demo.py

> The last default param 'times=1' are skipped over.



## Gamma

> Demo

```python
from pydpm._sampler import Basic_Sampler
import scipy.stats as stats

output = sampler.gamma(np.ones(1000)*4.5, 5)
plt.figure()
plt.hist(output, bins=20, density=True)
plt.plot(np.linspace(0, 100, 100), stats.gamma.pdf(np.linspace(0, 100, 100), 4.5, scale=5))
```

> Plot

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_gamma.png)



## Standard gamma

> Demo

```python
from pydpm._sampler import Basic_Sampler
import scipy.stats as stats

output = sampler.standard_gamma(np.ones(1000)*4.5)
plt.figure()
plt.hist(output, bins=20, density=True)
plt.plot(np.linspace(0, 20, 100), stats.gamma.pdf(np.linspace(0, 20, 100), 4.5))
```

> Plot

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_standard_gamma.png)



## Dirichlet

> Demo

```python
from pydpm._sampler import Basic_Sampler
import scipy.stats as stats

output = sampler.dirichlet(np.ones(1000)*4.5)
plt.figure()
plt.hist(output, bins=20, density=True)
```

> Plot

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_dirichlet.png)



## Beta

> Demo

```python
from pydpm._sampler import Basic_Sampler
import scipy.stats as stats

output = sampler.beta(np.ones(1000)*2, 5)
plt.figure()
plt.hist(output, bins=20, density=True)
plt.plot(np.linspace(0, 1, 100), stats.beta.pdf(np.linspace(0, 1, 100), 2, 5))
```

> Plot

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_beta.png)



## Normal

> Demo

```python
from pydpm._sampler import Basic_Sampler
import scipy.stats as stats

output = sampler.normal(np.ones(1000)*5, np.ones(1000)*2)
plt.figure()
plt.hist(output, bins=20, density=True)
plt.plot(np.linspace(-2, 13, 100), stats.norm.pdf(np.linspace(-2, 13, 100), 5, scale=2))
```

> Plot

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_normal.png)







## Binomial

> Demo

```python
from pydpm._sampler import Basic_Sampler
import scipy.stats as stats

output = sampler.binomial(np.ones(1000)*10, np.ones(1000)*0.5)
plt.figure()
plt.hist(output, bins=np.max(output)-np.min(output), density=True, range=(np.min(output)-0.5, np.max(output)-0.5))

```

> Plot

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_binomial.png)



## Poisson

> Demo

```python
from pydpm._sampler import Basic_Sampler
import scipy.stats as stats

output = sampler.poisson(np.ones(1000)*10)
plt.figure()
plt.hist(output, bins=22, density=True, range=(-0.5, 21.5))
plt.scatter(np.arange(20), stats.poisson.pmf(np.arange(20), 10), c='orange', zorder=10)
```

> Plot

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_poisson.png)



## Multinomial

> Demo

```python
from pydpm._sampler import Basic_Sampler
import scipy.stats as stats

output = sampler.multinomial(5, [0.8, 0.2], 1000)
plt.figure()
plt.hist(output[0], bins=15, density=True)
plt.title('multinomial(5, [0.8, 0.2])')
plt.show()

a = np.array([np.array([[i] * 6 for i in range(6)]).reshape(-1), np.array(list(range(6)) * 6)]).T
output = stats.multinomial(n=5, p=[0.8, 0.2]).pmf(a)
sns.heatmap(output.reshape(6, 6), annot=True)
plt.ylabel('number of the 1 kind(p=0.8)')
plt.xlabel('number of the 2 kind(p=0.2)')
plt.title('stats.multinomial(n=5, p=[0.8, 0.2])')
```

> Plot

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_multinomial.png)



