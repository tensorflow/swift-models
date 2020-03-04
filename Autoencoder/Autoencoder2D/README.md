# Simple 2D Autoencoder

This is an example of a simple 2-dimensional autoencoder model, using KuzushijiMNIST as a training dataset. It should produce output similar to the following:

### Epoch 1
<p align="center">
<img src="images/epoch-1-input.jpg" height="270" width="360">
<img src="images/epoch-1-output.jpg" height="270" width="360">
</p>

### Epoch 10
<p align="center">
<img src="images/epoch-10-input.jpg" height="270" width="360">
<img src="images/epoch-10-output.jpg" height="270" width="360">
</p>


## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```
swift run -c release Autoencoder2D
```
