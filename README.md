# LSQ-Net: Learned Step Size Quantization

## Introduction

This is an unofficial implementation of LSQ-Net, a deep neural network quantization framework.
LSQ-Net is proposed by Steven K. Esser and et al. from IBM. It can be found on [arXiv:1902.08153](https://arxiv.org/abs/1902.08153).

There are some little differences between my implementation and the original paper, which will be described in detail below.

If this repository is helpful to you, please star it.

## Results and Models

Here are some experiment results.
We will release more quantized models with different configurations soon. 

All these models can be downloaded from [Dropbox](https://www.dropbox.com/sh/un1k74qael1k6mx/AADroPMhvCrd1szG6HUYO_N3a?dl=0).

| Network   | Config. File | Model | Bitwidth (W/A) | Top-1 Acc. (%) | Top-5 Acc. (%) |
|:---------:|:--------:|:-------------:|:---------------:|:--------------:|:--------------:|
| ResNet-18 | [link](examples/lsq/resnet18_a2w3_imagenet.yaml) | [link](https://www.dropbox.com/sh/a5spn8boovfhjrj/AAD-Ureq7DpMKOujPdH4l0jVa?dl=0) |            3/2 |           66.9 |           87.2 |

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

The modified options in your YAML file will overwrite the default settings. For details, please read the comments in `config.yaml`. After every epoch, the program will automatically store the best model parameters as a checkpoint. You can modify the option `resume.path: /path/to/checkpoint.pth.tar` in the YAML file to resume the training process, or evaluate the accuracy of the quantized model.

You can find some example configuration files in the [example](examples) folder.

## Implementation Differences From the Original Paper

LSQ-Net paper has two versions, [v1](https://arxiv.org/pdf/1902.08153v2.pdf) and [v2](https://arxiv.org/pdf/1902.08153v1.pdf).
To improve accuracy, the authors expanded the quantization space in the v2 version. 
Recently they released a new version [v3](https://arxiv.org/pdf/1902.08153v3.pdf), which fixed some typos in the v2 version.

My implementation generally follows the v2 version, except for the following points.

### Initial Values of the Quantization Step Size

The authors use `Tensor(v.abs().mean() * 2 / sqrt(Qp))` as initial values of the step sizes in both weight and activation quantization layers, where Qp is the upper bound of the quantization space, and v is the initial weight values or the first batch of activations.

In my implementation, the step sizes in weight quantization layers are initialized in the same way, but in activation quantization layers, the step sizes are initialized as `Tensor(1.0)`.

### Supported Models

Currently, only ResNet is supported.
For the ImageNet dataset, the ResNet-18/34/50/101/152 models are copied from the torchvision model zoo. 
For the CIFAR10 dataset, the models are modified based on [Yerlan Idelbayev's contribution](https://github.com/akamaster/pytorch_resnet_cifar10), including ResNet-20/32/44/56/110/1202.

Thanks to the non-invasive nature of the framework, it is easy to add another new architectures beside ResNet.
All you need is to paste your model code into the `model` folder, and then add a corresponding entry in the `model/model.py`. 
The quantization framework will automatically replace layers specified in `quan/func.py` with their quantized versions automatically.

## Contributing Guide

I am not a professional algorithm researcher, and I only have very limited GPU resources. Thus, I may not spend too much time continuing to optimize its accuracy.

However, if you find any bugs in my code or have any ideas to improve the quantization results, please feel free to open an issue. I will be glad to join the discussion.
