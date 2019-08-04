# ResNet-50 with CIFAR-10

This example demonstrates how to train the [ResNet-50 network]( https://arxiv.org/abs/1512.03385) against the [CIFAR-10 image classification dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

A modified ResNet-50 network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the CIFAR-10 dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release ResNet-CIFAR10
```
