# ResNet-50

This directory contains a ResNet-50 image classification model for either ImageNet-size images
(224x224) or CIFAR-size images (32x32), and an example that trains this model on the CIFAR-10
dataset.

## Setup

You'll need the latest version of [Swift for TensorFlow][SwiftForTensorFlow]
installed and added to your path. Additionally, the data loader requires Python
3.x (rather than Python 2.7), `wget`, and `numpy`.

> Note: For MacOS, you need to set up the `PYTHON_LIBRARY` to help the Swift for
> TensorFlow find the `libpython3.<minor-value>.dylib` file, e.g., in `homebrew`.

To train the model on CIFAR-10, run:

```
cd swift-models
swift run ResNet
```

[SwiftForTensorFlow]: (https://github.com/tensorflow/swift/blob/master/Installation.md)
