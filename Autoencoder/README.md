# Simple Autoencoder

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

This directory builds a simple 1-dimensional autoencoder model.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```
swift run Autoencoder
```
If you using brew to install python2 and modules, change the path:
 - remove brew path '/usr/local/bin'
 - add TensorFlow swift Toolchain /Library/Developer/Toolchains/swift-latest/usr/bin

```
export PATH=/Library/Developer/Toolchains/swift-latest/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:"${PATH}"
``` 
