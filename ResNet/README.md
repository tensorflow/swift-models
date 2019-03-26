# ResNet-50

This directory contains a ResNet-50 image classification model for either ImageNet-size images
(224x224) or CIFAR-size images (32x32), and an example that trains this model on the CIFAR-10
dataset.

## Setup

You'll need the [latest version of Swift for TensorFlow]
(https://github.com/tensorflow/swift/blob/master/Installation.md)
installed and added to your path. Additionally, the data loader requires Python 3.x (rather than
Python 2.7), `wget`, and numpy.

To train the model on CIFAR-10, run:

```
swift -O ResNet50.swift data.swift main.swift
```
