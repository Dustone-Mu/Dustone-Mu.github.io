---
sort: 1
---

# Introduction

## About PyDPM

[pyDPM](https://github.com/BoChenGroup/pydpm)   [![PyPI](https://img.shields.io/pypi/v/gluonts.svg?style=flat-square)](https://pypi.org/project/gluonts/)   [![GitHub](https://img.shields.io/github/license/awslabs/gluon-ts.svg?style=flat-square)](./LICENSE)

Probabilistic topic models (PTMs) have been proven as an effective way to discover latent semantic structures from a collection of documents, where each document is represented as a
bag-of-words (BoW) vector, and been widely applied in the fields of machine learning (ML)
and natural language processing (NLP) in the past twenty years. 

With latent Dirchilet allocation (LDA) (Blei et al., 2001) being the best known representative, vanilla PTMs (Zhou et al., 2012; Gan et al., 2015; Zhou et al., 2015) and their sophisticated extensions can not only discover a set of underlying topics from the raw text corpus, where each topic describes an interpretable semantic concept, but also provide low-dimensional latent document representations, which can be directly applied for downstream tasks.


Although most of the aforementioned PTMs have released their source codes, these published projects are developed in a lot of various programming languages, like C (Ahmed et al., 2012), Matlab (Steyvers and Griffiths, 2011) and Java (McCallum, 2002; Ramage and Rosen, 2011) etc. And as far as we known, there is still no unified probabilistic library to collect or even summarize these PTMs under the same codebase. 

To this end, we present the **PyDPM** library, a Python package that has contained a wide variety of [PTMs](https://dustone-mu.github.io/Model/), each of which has been reimplemented under a well-designd codebase and can be directly called from the library as an API. The pyDPM library also provides a [self-developed Sampler module](https://dustone-mu.github.io/Sample/), whose GPU version can greatly improve the sampling efficiency through CUDA’s parallel computing, and a [Metric module](https://dustone-mu.github.io/Metric/) to evaluate the performance of these PTMs after training. In the following sections, we will exhaustively describe the main features of the pyDPM library, instructions for installation and usage.

## Software Description

[PyDPM](https://github.com/BoChenGroup/pydpm) is a Python Library for Deep Probabilistic Models

PyDPM focuses on constructing deep probabilistic models on GPU. It provides efficient distribution sampling functions and has included lots of implemented probabilistic models.

As shown in Fig. 1, the pyDPM library can be roughly split into four modules, which are Sampler, Model, Metric and Example modules, respectively. 

Generally speaking, the Sampler module provides both the most basic distribution sampler and model sampler for training or testing the constructed PTMs on CPU or GPU; the Model module contains a wide variety of popular PTMs, which can be directly called as APIs in Python; the Metric module includes a series of widely used topic modeling metrics to evaluate these PTMs; For each topic model included in the Model module, the Example module provides a demo code equipped a detailed tutorial about how to use and evaluate this topic model.

![Image text](https://raw.githubusercontent.com/BoChenGroup/pydpm/master/pydpm_framework.png "Figure 1: The whole framework of the developed pyDPM library, including Sampler, Model, Metric and Example modules.")

