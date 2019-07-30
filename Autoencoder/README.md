# Simple Autoencoder

This is an example of a simple 1-dimensional autoencoder model, using MNIST as a training dataset. It should produce output similar to the following:

### Epoch 1
<p align="center">
<img src="images/epoch-1-input.png" height="270" width="360">
<img src="images/epoch-1-output.png" height="270" width="360">
</p>

### Epoch 10
<p align="center">
<img src="images/epoch-10-input.png" height="270" width="360">
<img src="images/epoch-10-output.png" height="270" width="360">
</p>


## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

This example requires Matplotlib and NumPy to be installed, for use in image output.

To train the model, run:

```
swift run -c release Autoencoder
```
