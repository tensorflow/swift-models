# CIFAR-10 with custom models

This example demonstrates how to train the custom-defined models (based on examples from [PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and [Keras](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py) ) against the [CIFAR-10 image classification dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

Two custom models are defined, and one is applied to an instance of the CIFAR-10 dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/main/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release Custom-CIFAR10
```
