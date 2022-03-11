---
sort: 4
---

# Model

{% include list.liquid %}


Create Probabilistic Model
=============

>Model list
>
Model list is as following:

|Probabilistic Model Name                  |Abbreviation |Paper Link|
|------------------------------------------|-------------|----------|
|Latent Dirichlet Allocation               |LDA          |[Link](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)|
|Poisson Factor Analysis                   |PFA          |[Link](http://mingyuanzhou.github.io/Papers/AISTATS2012_NegBinoBeta_PFA_v19.pdf)|
|Poisson Gamma Belief Network              |PGBN         |[Link](http://mingyuanzhou.github.io/Papers/DeepPoGamma_v5.pdf )|
|Convolutional Poisson Factor Analysis     |CPFA         |[Link](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf)|
|Convolutional Poisson Gamma Belief Network|CPGBN        |[Link](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf)|
|Poisson Gamma Dynamical Systems           |PGDS         |[Link](http://mingyuanzhou.github.io/Papers/ScheinZhouWallach2016_paper.pdf )|
|Deep Poisson Gamma Dynamical Systems      |DPGDS        |[Link](http://mingyuanzhou.github.io/Papers/Guo_DPGDS_NIPS2018.pdf)|
|Dirichlet Belief Networks                 |DirBN        |[Link](https://arxiv.org/pdf/1811.00717.pdf)|
|Deep Poisson Factor Analysis              |DPFA         |[Link](http://proceedings.mlr.press/v37/gan15.pdf)|
|Word Embeddings Deep Topic Model          |WEDTM        |[Link](http://proceedings.mlr.press/v80/zhao18a/zhao18a.pdf)|

More probabilistic models will be further included in pydpm/_model/...

>Demo

Create a PGBN model:
```python
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

# save the model after training
model.save()
```
More model demos can be found in pydpm/examples/...

Source data can be found in [Link](https://drive.google.com/drive/folders/1_BH_0N6wfbUvTS-CCWs4YLFpDWqGRw7w?usp=sharing)