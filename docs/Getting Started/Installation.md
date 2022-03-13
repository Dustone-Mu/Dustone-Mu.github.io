---
sort: 2
---

# Installation

## GitHub&PyPi

This library can be installed under both Windows and Linux systems.

Package can be cloned from [GitHub](https://github.com/BoChenGroup/pydpm) or installed by [PyPi](https://pypi.org/project/pydpm/).

```
$ pip install pydpm
```

## Requirements

Pydpm is written in python 3 and accelerated by CUDA C++.

To achieve the best performance of the distribution sampling functions and the inference of probabilistic models, a GPU and NVIDIA CUDA Toolkit are necessary.

Other requirements of python packages are as following:

> numpy
> scipy
> sklearn
> pycuda
> gensim

!!! tip
    Under Windows system, we recommed to install Visual Studio 2019 and latest CUDA Toolkit. The combination of VS2019(with MSVC v142) and CUDA 11.5 has been tested in pydpm2.0.
