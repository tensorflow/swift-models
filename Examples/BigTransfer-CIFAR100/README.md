# Big Transfer with CIFAR-100

This script illustrates how to train Big Transfer (https://arxiv.org/abs/1912.11370) against the [CIFAR-100 image classification dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

This model uses a pre-defined rule based on dataset size to determine the optimal parameters for fine tuning using a slightly modified ResnetV2 transfer learning model.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/main/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run BigTransfer-CIFAR100
```
