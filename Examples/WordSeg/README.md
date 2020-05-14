# Word Segmentation

This example demonstrates how to train the [word segmentation model][paper]
against the dataset provided in the paper.

A segmental neural language model (SNLM) is instantiated from the library of
standard models. A custom training loop is defined and the training and validation
losses and accuracies for each epoch are shown during training.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release WordSeg
```

[paper]: https://www.aclweb.org/anthology/P19-1645.pdf
