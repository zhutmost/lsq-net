# LSQ-Net: Learned Step Size Quantization

## Introduction

This is an unofficial implementation of LSQ-Net, a deep neural network quantization framework.
LSQ-Net is proposed by Steven K. Esser and et al. from IBM. It can be found on [arXiv:1902.08153](https://arxiv.org/abs/1902.08153).

Due to the outbreak of coronavirus pneumonia in China, I cannot return to my laboratory for the time being, so several latest commits and experimental results have not been pushed to this repository. Currently, my implementation can obtain the same accuracy as in the original paper.

There are some little differences between my implementation and the original paper, which will be described in detail below.

If this repository is helpful to you, please star it.

## User Guide

### Install Dependencies

First install library dependencies within an Anaconda environment.

```bash
# Create a environment with Python 3.8
conda create -n lsq python=3.8
# PyTorch GPU version >= 1.5
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# Tensorboard visualization tool
conda install tensorboard
# Miscellaneous
conda install scikit-learn pyyaml munch
```

### Run Scripts with Your Configurations

This program use YAML files as inputs. A template as well as the default configuration is providen as `config.yaml`.

If you want to change the behaviour of this program, please copy it somewhere else. And then run the `main.py` with your modified configuration file.

```
python main.py /path/to/your/config/file.yaml
```

The modified options in your YAML file will overwrite the default settings. For details, please read the comments in `config.yaml`.

After every epoch, the program will automatically store the best model parameters as a checkpoint. You can modify the option `resume.path: /path/to/checkpoint.pth.tar` in the YAML file to resume the training process, or evaluate the accuracy of the quantized model.

## Implementation Differences From the Original Paper

LSQ-Net paper has two versions, [v1](https://arxiv.org/pdf/1902.08153v2.pdf) and [v2](https://arxiv.org/pdf/1902.08153v1.pdf).
To improve accuracy, the authors expanded the quantization space in the v2 version. 
Recently they released a new version [v3](https://arxiv.org/pdf/1902.08153v3.pdf), which fixed some typos in the v2 version.

My implementation generally follows the v2 version, except for the following points.

### Quantization of the First and Last Layers

The authors quantize the first convolution layer and the last fully-connected layer to 8-bit fixed-point numbers. However, their description is not clear. Due to the normalization in the input image preprocessing stage, the input of the first layer may be negative, so it cannot be applied to the activation quantizer in the original paper.
Therefore, please handle the first layer carefully. In my case, I put it in `quan.excepts` to avoid quantization.

### Initial Values of the Quantization Step Size

The authors use 2<|v|>/sqrt(Qp) as initial values of the step sizes in both weight and activation quantization layers, where Qp is the upper bound of the quantization space, and v is the initial weight values or the first batch of activations.

In my implementation, the step sizes in weight quantization layers are initialized as `Tensor(v.abs().mean()/Qp)`. In activation quantization layers, the step sizes are initialized as `Tensor(1.0)`.

### Optimizers and Hyper-parameters

In the original paper, the network parameters are updated by a SGD optimizer with a momentum of 0.9, a weight decay of 10^-5 ~ 10^-4 and a initial learning rate of 0.01.
A cosine learning rate decay without restarts is also performed to adjust the learning rate during training.

I also use a SGD optimizer, but the weight decay is fixed to 10^-4. A step scheduler is used to decrease the learning rate 10 times every 30 epoch.

All the configurable hyper-parameters can be found in the YAML configuration file.

### Supported Models

Currently, only ResNet-18/34/50/101/152 is supported, because I do not have enough GPUs to evaluate my code on other networks. Nevertheless, it is easy to add another new architecture beside ResNet.

All you need is to extend the `Quantize` class in `quan/lsq.py`. With it, you can easily insert activation/weight quantization layers before matrix multiplication in your networks. Please refer to the implementation of the `QuanConv2d` class.

## Contributing Guide

I am not a professional algorithm researcher, and the current accuracy is enough for me. And I only have very limited GPU resources. Thus, I may not spend too much time continuing to optimize its results.

However, if you find any bugs in my code or have any ideas to improve the quantization results, please feel free to open an issue. I will be glad to join the discussion.
