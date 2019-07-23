# ResNet-50

This directory contains different ResNet image classification models for either ImageNet-size images
(224x224) or CIFAR-size images (32x32), and an example that trains a ResNet-50 model on the CIFAR-10
dataset.

## Setup

You'll need [the latest version][INSTALL] of Swift for TensorFlow
installed and added to your path.

To train the model on CIFAR-10, run:

```
cd swift-models
swift run -c release ResNet
```

[INSTALL]: (https://github.com/tensorflow/swift/blob/master/Installation.md)
