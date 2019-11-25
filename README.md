# LSQ-Net: Learned Step Size Quantization

## Introduction

This is an unofficial implementation of LSQ-Net, a deep neural network quantization framework.
LSQ-Net is proposed by Steven K. Esser and et al. from IBM. It can be found on [arXiv:1902.08153](https://arxiv.org/abs/1902.08153).

|Model|Top-1 @ Original|Top-1 @ This work|Top-5 @ Original|Top-5 @ This work|
|:----------:|:----:|:----:|:----:|:---:|
| ResNet-18  | 70.2 |      | 89.4 |     |
| ResNet-34  | 70.2 |      | 89.4 |     |
| ResNet-50  | 70.2 |      | 89.4 |     |
| ResNet-101 | 70.2 |      | 89.4 |     |

There are some little differences between my implementation and the original paper, which will be described in detail below.

If this repository is helpful to you, please star it.

## User Guide

Here are some examples to show the usage of this project.

1. Evaluate a quantized ResNet-18 model with the ImageNet dataset.
```
python main.py --dataset-dir /path/to/imagenet --gpu 0,1 --name proj_name --load-workers 16 --arch resnet18 --resume /path/to/checkpoint.pth.tar --eval --quan-bit-a 3 --quan-bit-w 3
```
2. Quantize a pre-trained ResNet-18 to 3-bit weights/activations.
```
python main.py --dataset-dir /path/to/imagenet --gpu 0,1 --name proj_name --load-workers 16 --arch resnet18 --pre-trained --quan-bit-a 3 --quan-bit-w 3 --weight-decay 0.00005
```

After every epoch, the program will automatically store the best model parameters as a checkpoint. You can use the argument `--resume /path/to/checkpoint.pth.tar` to resume the training process, or evaluate the accuracy of the quantized model.

For detailed argument usage, please use `python main.py -h` for help, or read `util/config.py` directly.


## Implementation Differences From the Original Paper

LSQ-Net paper has two versions, [v1](https://arxiv.org/pdf/1902.08153v2.pdf) and [v2](https://arxiv.org/pdf/1902.08153v1.pdf). To improve accuracy, the authors modified the quantization method in the v2 version, mainly including expanding the quantization space.

My implementation generally follows the v2 version, except for the following points.

### Initial Values of the Quantizer Step Size

The authors use 2<|v|>/sqrt(Qp) as initial values of the step sizes in both weight and activation quantization layers, where Qp is the upper bound of the quantization space, and v is the initial weight values or the first batch of activations.

In my implementation, the step sizes in weight quantization layers are initialzed as `Tensor(v.abs().mean()/Qp)`. In activation quantization layers, the step sizes are initialized as `Tensor(1.0)`.

### Supported Models

Currently, only ResNet-18/34/50/101 is supported, because I do not have enough GPUs to evaluate my code on other networks. Nevertheless, it is easy to add another new architecture beside ResNet.

All you need is a `Quantize` class in `quan/lsq.py`. With it, you can easily insert activation/weight quantization layers before matrix multiplication in your networks.

## Contributing Guide

I am not a professional algorithm researcher, and the current accuracy is enough for me. And I only have very limited GPU resources. Thus, I may not spend too much time continuing to optimize the code.

However, if you find any bugs in my code or have any ideas to improve the quantization results, please feel free to open an issue. I will be glad to join the discussion.
