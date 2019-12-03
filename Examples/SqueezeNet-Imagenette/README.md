# SqueezeNet with Imagenette

This example demonstrates how to train the [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf) network against the [Imagenette image classification dataset](https://github.com/fastai/imagenette).

A SqueezeNet (version 1.1) network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the Imagenette dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release SqueezeNet-Imagenette
```
