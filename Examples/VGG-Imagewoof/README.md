# VGG with Imagewoof

This example demonstrates how to train the [VGG](https://arxiv.org/abs/1409.1556) network against the [Imagewoof image classification dataset](https://github.com/fastai/imagenette) (a harder version of Imagenette).

A VGG-16 network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the Imagewoof dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.

As a note: the current implementation of the Imagewoof dataset loads all images into memory as floats, which can lead to memory exhaustion on machines with less than 16 GB of available RAM.

## Learning rate schedules

VGG, as a network, is extremely sensitive to training tweaks and batch size.  This code sample demonstrates using a custom decreasing learning rate schedule, similar to that employed in the original paper.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release VGG-Imagewoof
```
