---
sort: 1
---

# Introduction

The Model module in pyDPM has included a wide variety of popular PTMs, which can be roughly split into serveral categories: 1) basic topic models;2) deep topic models; 3) sequential topic models; 4) topic model based extensions.

All models are as following:

<!--
1) basic topic models, 
	like Latent Dirichlet Allocation (LDA) (Blei et al., 2001) and 
	Poisson Factor Analysis (PFA) (Zhou et al., 2012); 

2) deep topic models, like 
	Deep Poisson Factor Analysis (DPFA) (Gan et al., 2015), 
	Poisson Gamma Belief Network (PGBN) (Zhou et al., 2015), 
	Dirichlet Belief Networks (DirBN) (Zhao et al., 2018b) and 
	Word Embeddings Deep Topic Model (WEDTM) (Zhao et al., 2018a); 

3) sequential topic models, like 
	Convolutional Poisson Factor Analysis (CPFA) (Wang et al., 2019), 
	Convolutional Poisson Gamma Belief Network (CPGBN) (Wang et al., 2019), 
	Poisson Gamma Dynamical Systems (PGDS) (Schein et al., 2016) and 
	Deep Poisson Gamma Dynamical Systems (DPGDS) (Guo et al., 2018); 

4) topic model based extensions, like 
	Multimodal Poisson gamma belief network (MPGBN) (Wang et al., 2018) and 
	Graph Poisson gamma belief network (GPGBN) (Wang et al., 2020).
-->

> Probabilistic model list

|Type	            |Probabilistic Model Name                  |Abbreviation |Paper Link|
|-------------------|------------------------------------------|-------------|----------|
|Basic TM           |Latent Dirichlet Allocation               |LDA          |[Link](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)|
|Basic TM           |Poisson Factor Analysis                   |PFA          |[Link](http://mingyuanzhou.github.io/Papers/AISTATS2012_NegBinoBeta_PFA_v19.pdf)|
|Deep TM            |Poisson Gamma Belief Network              |PGBN         |[Link](http://mingyuanzhou.github.io/Papers/DeepPoGamma_v5.pdf )|
|Deep TM            |Deep Poisson Factor Analysis              |DPFA         |[Link](http://proceedings.mlr.press/v37/gan15.pdf)|
|Deep TM            |Dirichlet Belief Networks                 |DirBN        |[Link](https://arxiv.org/pdf/1811.00717.pdf)|
|Deep TM            |Word Embeddings Deep Topic Model          |WEDTM        |[Link](http://proceedings.mlr.press/v80/zhao18a/zhao18a.pdf)|
|Sequential TM      |Convolutional Poisson Factor Analysis     |CPFA         |[Link](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf)|
|Sequential TM      |Convolutional Poisson Gamma Belief Network|CPGBN        |[Link](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf)|
|Sequential TM      |Poisson Gamma Dynamical Systems           |PGDS         |[Link](http://mingyuanzhou.github.io/Papers/ScheinZhouWallach2016_paper.pdf)|
|Sequential TM      |Deep Poisson Gamma Dynamical Systems      |DPGDS        |[Link](http://mingyuanzhou.github.io/Papers/Guo_DPGDS_NIPS2018.pdf)|
|TM based extensions|Multimodal Poisson Gamma Belief Network   |MPGBN        |[Link](https://mingyuanzhou.github.io/Papers/mpgbn_aaai18.pdf)|
|TM based extensions|Graph Poisson Gamma Belief Network        |GPGBN        |[Link](https://proceedings.neurips.cc/paper/2020/file/05ee45de8d877c3949760a94fa691533-Paper.pdf)|


More probabilistic models will be further included in pydpm/_model/...

