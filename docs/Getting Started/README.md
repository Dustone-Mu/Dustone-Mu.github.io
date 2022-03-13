---
sort: 1
---

# Getting Started

Topic modeling is a frequently used text-mining tool and has achieved great success in the field of text analysis. In the past two decades, a wide variety of popular probabilistic topic models (PTMs) have been developed, but there is still no unified library to collect or even summarize these topic models under the same codebase as far as we known.

To this end, we introduce the **pyDPM**, an open-source Python library that provides a series of reimplementations of these popular PTMs. The pyDPM library also includes a self-developed Sampler module, which can efficiently process Gibbs sampling to train the constructed PTMs deployed on either CPU or GPU, and a Metric module to evaluate the performance of these PTMs on downstream tasks. 

The source code has been released at the following Github address: [https://github.com/BoChenGroup/pydpm](https://github.com/BoChenGroup/pydpm).

> Quick start:

{% include list.liquid %}
