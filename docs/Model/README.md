---
sort: 4
---

# Model

{% include list.liquid %}

## Introduction

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


> In the following part, we introduce All models' demo. The source data used can be found in [Google drive](https://drive.google.com/drive/folders/1_BH_0N6wfbUvTS-CCWs4YLFpDWqGRw7w?usp=sharing)


## LDA

Latent Dirichlet Allocation(LDA)

[Latent dirichlet allocation. David M. Blei, Andrew Y. Ng, and Michael I. Jordan.](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) In *Advances in Neural Information Processing Systems.*

> Demo

```python
from pydpm._model import LDA

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]

# create the model and deploy it on gpu or cpu
model = LDA(128, 'gpu')
model.initial(train_data)  # use the shape of train_data to initialize the params of model
train_local_params = model.train(100, train_data)
train_local_params = model.test(100, train_data)
test_local_params = model.test(100, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.850
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
```


## PFA

Poisson Factor Analysis(PFA)

[Beta-negative binomial process and poisson factor analysis. Mingyuan Zhou, Lauren Hannah, David B. Dunson, and Lawrence Carin.](http://mingyuanzhou.github.io/Papers/AISTATS2012_NegBinoBeta_PFA_v19.pdf) In *International Conference on Artificial Intelligence and Statistics.*

> Demo

```python
from pydpm._model import PFA

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]

# create the model and deploy it on gpu or cpu
model = PFA(128, 'gpu')
model.initial(train_data)  # use the shape of train_data to initialize the params of model
train_local_params = model.train(100, train_data)
train_local_params = model.test(100, train_data)
test_local_params = model.test(100, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8238
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
```


## PGBN

Poisson Gamma Belief Network(PGBN)

[The poisson gamma belief network. Mingyuan Zhou, Yulai Cong, and Bo Chen.](http://mingyuanzhou.github.io/Papers/DeepPoGamma_v5.pdf) In *Advances in Neural Information Processing.*

> Demo

```python
from pydpm._model import PGBN

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]

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


## DPFA

Deep Poisson Factor Analysis(DPFA)

[Scalable deep poisson factor analysis for topic modeling. Zhe Gan, Changyou Chen, Ricardo Henao, David E. Carlson, and Lawrence Carin.](http://proceedings.mlr.press/v37/gan15.pdf) In *International Conference on Machine Learning.*

> Demo

```python
from pydpm._model import DPFA

# load data
data = sio.loadmat('./data/mnist_gray.mat')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]

# create the model and deploy it on gpu or cpu
model = DPFA([128, 64, 32], 'gpu')  # topics of 3 layers
model.initial(train_data)  # use the shape of train_data to initialize the params of model
burnin, collection = 100, 80
train_local_params = model.train(burnin, collection, train_data)
train_local_params = model.test(burnin, collection, train_data)
test_local_params = model.test(burnin, collection, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.9099
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
```


## DirBN

Dirichlet Belief Networks(DirBN)

[Dirichlet belief networks for topic structure learning. He Zhao, Lan Du, Wray L. Buntine, and Mingyuan Zhou.](https://arxiv.org/pdf/1811.00717.pdf) In *Advances in Neural Information Processing Systems.*

> Demo

```python
from pydpm._model import DirBN

# load data
data = sio.loadmat('./data/mnist_gray.mat')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:499]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1499]
train_label = data['train_label'][:499]
test_label = data['train_label'][1000:1499]

# create the model and deploy it on gpu or cpu
model = DirBN([100, 100], 'gpu')  # topics of each layers
model.initial(train_data)  # use the shape of train_data to initialize the params of model
train_local_params = model.train(90, train_data)
train_local_params = model.test(90, train_data)
test_local_params = model.test(90, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.78
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
```


## WEDTM

Word Embeddings Deep Topic Model(WEDTM)

[Inter and intra topic structure learning with word embeddings. He Zhao, Lan Du, Wray L. Buntine, and Mingyuan Zhou.](http://proceedings.mlr.press/v80/zhao18a/zhao18a.pdf) In *International Conference on Machine Learning.*

> Demo

```python
from pydpm._model import WEDTM

# load data
dataset = sio.loadmat('./data/WS.mat')
train_data = np.asarray(dataset['doc'].todense()[:, dataset['train_idx'][0]-1])[:, ::10].astype(int)
test_data = np.asarray(dataset['doc'].todense()[:, dataset['test_idx'][0]-1])[:, ::5].astype(int)
train_label = dataset['labels'][dataset['train_idx'][0]-1][::10, :]
test_label = dataset['labels'][dataset['test_idx'][0]-1][::5, :]

# params of model
T = 3  # vertical layers
S = 3  # sub topics
K = [100] * T  # topics in each layers

# create the model and deploy it on gpu or cpu
model = WEDTM(K, 'gpu')
model.initial(dataset['doc'])  # use the shape of train_data to initialize the params of model
train_local_params = model.train(dataset['embeddings'], S, 300, train_data)
train_local_params = model.test(dataset['embeddings'], S, 300, train_data)
test_local_params = model.test(dataset['embeddings'], S, 300, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
```


## CPFA

Convolutional Poisson Factor Analysis(CPFA)

[Convolutional poisson gamma belief network. Chaojie Wang, Bo Chen, Sucheng Xiao, and Mingyuan Zhou.](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf) In *International Conference on Machine Learning.*

> Demo

```python
from pydpm._model import CPFA

DATA = cPickle.load(open("./data/TREC.pkl", "rb"), encoding='iso-8859-1')
# ========== details of data process can be found in pydpm/example/CPFA_demo.py =========

# create the model and deploy it on gpu or cpu
model = CPFA(200, 'gpu')
# mode 1, dense input
model.initial([batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])  # use the shape of train_data to initialize the params of model
train_local_params = model.train(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
train_local_params = model.test(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
test_local_params = model.test(100, [batch_file_index_te, batch_rows_te, batch_cols_te, batch_value_te], [len(data_test_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_te)])

train_theta = np.sum(np.sum(train_local_params.W_nk, axis=3), axis=2).T
test_theta = np.sum(np.sum(test_local_params.W_nk, axis=3), axis=2).T

# train_theta[np.where(np.isinf((train_theta)))] = 0

# Score of test dataset's Theta: 0.682
results = ACC(train_theta, test_theta, batch_label_tr, batch_label_te, 'SVM')
model.save()
```


## CPGBN

Convolutional Poisson Gamma Belief Network(CPGBN)

[Convolutional poisson gamma belief network. Chaojie Wang, Bo Chen, Sucheng Xiao, and Mingyuan Zhou.](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf) In *International Conference on Machine Learning.*

> Demo

```python
from pydpm._model import CPGBN

DATA = cPickle.load(open("data/TREC.pkl", "rb"), encoding='iso-8859-1')
# ========== details of data process can be found in pydpm/example/CPFA_demo.py =========

# create the model and deploy it on gpu or cpu
model = CPGBN([200, 100, 50], 'gpu')
# mode 1, dense input
model.initial([batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])  # use the shape of train_data to initialize the params of model
train_local_params = model.train(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
train_local_params = model.test(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
test_local_params = model.test(100, [batch_file_index_te, batch_rows_te, batch_cols_te, batch_value_te], [len(data_test_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_te)])

train_theta = np.sum(np.sum(train_local_params.W_nk, axis=3), axis=2).T
test_theta = np.sum(np.sum(test_local_params.W_nk, axis=3), axis=2).T

# Score of test dataset's Theta: 0.682
results = ACC(train_theta, test_theta, batch_label_tr, batch_label_te, 'SVM')
model.save()
```


## PGDS

Poisson Gamma Dynamical Systems(PGDS)

[Poisson-gamma dynamical systems. Aaron Schein, Hanna M. Wallach, and Mingyuan Zhou.](http://mingyuanzhou.github.io/Papers/ScheinZhouWallach2016_paper.pdf) In *Advances in Neural Information Processing Systems.*

> Demo

```python
from pydpm._model import PGDS

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]

# create the model and deploy it on gpu or cpu
model = PGDS(100, 'gpu')
model.initial(train_data)
train_local_params = model.train(200, train_data)
train_local_params = model.test(200, train_data)
test_local_params = model.test(200, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8739
results = ACC(train_local_params.Theta, test_local_params.Theta, train_label, test_label, 'SVM')

# save the model after training
model.save()
```


## DPGDS

Deep Poisson Gamma Dynamical Systems(DPGDS)

[Deep poisson gamma dynamical systems. Dandan Guo, Bo Chen, Hao Zhang, and Mingyuan Zhou.](http://mingyuanzhou.github.io/Papers/Guo_DPGDS_NIPS2018.pdf) In *Advances in Neural Information Processing Systems.*

> Demo

```python
from pydpm._model import DPGDS

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]

# create the model and deploy it on gpu or cpu
model = DPGDS([200, 100, 50], 'gpu')
model.initial(train_data)
train_local_params = model.train(200, train_data)
train_local_params = model.test(200, train_data)
test_local_params = model.test(200, test_data)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8519
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_label, test_label, 'SVM')

# save the model after training
model.save()
```


## MPGBN

Multimodal Poisson Gamma Belief Network(MPGBN)

[Multimodal poisson gamma belief network. Chaojie Wang, Bo Chen, and Mingyuan Zhou.](https://mingyuanzhou.github.io/Papers/mpgbn_aaai18.pdf) In *AAAI Conference on Artificial Intelligence.*

> Demo

```python
from pydpm._model import MPGBN

# load data
data = sio.loadmat('./data/mnist_gray')
train_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 0:999]
train_data_1 = train_data[:360, :]
train_data_2 = train_data[360:, :]

test_data = np.array(np.ceil(data['train_mnist']*5), order='C')[:, 1000:1999]
test_data_1 = test_data[:360, :]
test_data_2 = test_data[360:, :]

train_label = data['train_label'][:999]
test_label = data['train_label'][1000:1999]


# create the model and deploy it on gpu or cpu
model = MPGBN([128, 64, 32], device='gpu')
model.initial(train_data_1, train_data_2)
train_local_params = model.train(100, train_data_1, train_data_2)
train_local_params = model.test(100, train_data_1, train_data_2)
test_local_params = model.test(100, test_data_1, test_data_2)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8659 -
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_label, test_label, 'SVM')

# save the model after training
model.save()
```


## GPGBN

Graph Poisson Gamma Belief Network(GPGBN)

[Deep relational topic modeling via graph poisson gamma belief network. Chaojie Wang, Hao Zhang, Bo Chen, Dongsheng Wang, Zhengjue Wang, and Mingyuan Zhou.](https://proceedings.neurips.cc/paper/2020/file/05ee45de8d877c3949760a94fa691533-Paper.pdf) In *Advances in Neural Information Processing Systems.*

> Demo

```python
from pydpm._model import CPGBN

DATA = cPickle.load(open("data/TREC.pkl", "rb"), encoding='iso-8859-1')
# ========== details of data process can be found in pydpm/example/CPFA_demo.py =========

# create the model and deploy it on gpu or cpu
model = CPGBN([200, 100, 50], 'gpu')
# mode 1, dense input
model.initial([batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])  # use the shape of train_data to initialize the params of model
train_local_params = model.train(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
train_local_params = model.test(100, [batch_file_index_tr, batch_rows_tr, batch_cols_tr, batch_value_tr], [len(data_train_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_tr)])
test_local_params = model.test(100, [batch_file_index_te, batch_rows_te, batch_cols_te, batch_value_te], [len(data_test_list) - delete_count, DATA['Vab_Size'], np.max(batch_len_te)])

train_theta = np.sum(np.sum(train_local_params.W_nk, axis=3), axis=2).T
test_theta = np.sum(np.sum(test_local_params.W_nk, axis=3), axis=2).T

# Score of test dataset's Theta: 0.682
results = ACC(train_theta, test_theta, batch_label_tr, batch_label_te, 'SVM')
model.save()
```



