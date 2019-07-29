# LeNet-5 with MNIST

This example demonstrates how to train the [LeNet-5 network]( http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) against the [MNIST digit classification dataset](http://yann.lecun.com/exdb/mnist/).

The LeNet network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the MNIST dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.


## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release LeNet-MNIST
```
