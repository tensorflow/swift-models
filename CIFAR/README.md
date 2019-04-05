# CIFAR

This directory contains different example convolutional networks for image
classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Setup

You'll need the latest version of [Swift for TensorFlow][SwiftForTensorFlow]
installed and added to your path. Additionally, the data loader requires Python
3.x (rather than Python 2.7), `wget`, and `numpy`.

> Note: For MacOS, you need to set up the `PYTHON_LIBRARY` to help the Swift for
> TensorFlow find the `libpython3.<minor-value>.dylib` file, e.g., in `homebrew`.

To train the default model, run:

```
cd swift-models
swift run -c release CIFAR
```

[SwiftForTensorFlow]: (https://github.com/tensorflow/swift/blob/master/Installation.md)
