# CIFAR

This directory contains two example convolutional networks for image classification on the
CIFAR-10 dataset.

## Setup

You'll need the [latest version of Swift for TensorFlow]
(https://github.com/tensorflow/swift/blob/master/Installation.md)
installed and added to your path. Additionally, the data loader requires Python 3.x (rather than
Python 2.7), `wget`, and numpy.

To train the default model, run:

```
cat models.swift data.swift main.swift | swift -O -
```
