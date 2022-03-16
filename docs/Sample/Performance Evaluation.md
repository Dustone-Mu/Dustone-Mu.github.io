---
sort: 2
---

# Performance Evaluation

## Compare with other libraries

The sampling efficiency comparisons between our Sampler module and other libraries under the same sampling conditions have been exhibited below

![Image text](https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_numpy.png "Compare with numpy")  
The compared code can be found in pydpm/example/Sampler_Speed_Demo.py

Compare the sampling speed of distribution functions with tensorflow and torch:
![Image text](https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_tf2_torch.png "Compare with TensorFlow and Torch")  
The compared code can be found in pydpm/example/Sampler_Speed_Demo.py


## An Evaluation demo

In order to verify the accuracy of the sampler function, comparsions between the distribution of the sampling results and the actual distribution function are implemented. The results are as following

![Image text](https://raw.githubusercontent.com/Dustone-Mu/Dustone-Mu.github.io/main/images/sample_demo_gamma.png)

The integral comparsion of sampler results and standard distribution can be found in pydpm/_example/Sampler_Demo.py and [*Distribution Sampler*](https://dustone-mu.github.io/Sample/Distribution%20Sampler.html) part.



